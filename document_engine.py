import os
import sys
from typing import List
import io
import msoffcrypto # Uncomment if needed for encrypted docs
import openpyxl # Uncomment if needed explicitly, though used by pandas/loaders usually

# Concurrency imports
import concurrent.futures
import multiprocessing

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START

from fritz_utils import CHROMA_DB_PATH, INDEXED_FILES_PATH, CHROMA_COLLECTION_NAME, DOC_FOLDER, FAST_OLLAMA_MODEL, \
    THINKING_OLLAMA_MODEL, EMBEDDING_MODEL

# Define supported file extensions
SUPPORTED_EXTENSIONS = ('.docx', '.pdf', '.xlsx', '.csv', '.txt', '.md')

# We will store the initialized retriever here so we don't rebuild it per query
GLOBAL_RETRIEVER = None

# --- OCR SETUP ---
try:
    from PIL import Image
    import easyocr

    OCR_AVAILABLE = True
    # Initialize EasyOCR reader (done once)
    # Note: In multiprocessing, the reader needs to be initialized per process or handled carefully.
    # We will initialize it lazily in the function to be safe across OS types.
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: EasyOCR not installed. OCR fallback will not be available.")

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Cannot render PDF pages for OCR.")


def get_ocr_reader():
    """Lazy loader for EasyOCR to ensure it works in subprocesses."""
    if OCR_AVAILABLE:
        # gpu=False is safer for multiprocessing to avoid CUDA context conflicts
        return easyocr.Reader(['en'], gpu=False)
    return None


# --- DOCUMENT LOADING FUNCTIONS ---

def load_pdf_with_ocr_fallback(file_path: str) -> List[Document]:
    """
    Loads a PDF file. First tries PyPDFLoader for embedded text.
    If text is empty or too short, falls back to OCR.
    """
    try:
        # Try standard PDF loading first
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if we got meaningful text
        total_text = "".join([doc.page_content for doc in docs]).strip()
        if len(total_text) > 50:
            return docs

        # Fall back to OCR
        if not OCR_AVAILABLE or not PYMUPDF_AVAILABLE:
            return docs

        # Initialize reader locally for this process
        reader = get_ocr_reader()
        if not reader:
            return docs

        pdf_document = fitz.open(file_path)
        ocr_docs = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            result = reader.readtext(img_byte_arr, detail=0)
            text = "\n".join(result)

            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": page_num,
                    "extraction_method": "ocr_easyocr"
                }
            )
            ocr_docs.append(doc)

        pdf_document.close()
        return ocr_docs

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []


def load_document_by_extension(file_path: str) -> List[Document]:
    """
    Selects the appropriate loader based on file extension.
    Top-level function for pickling support in multiprocessing.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return load_pdf_with_ocr_fallback(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            return loader.load()
        elif ext == '.csv':
            loader = CSVLoader(file_path)
            return loader.load()
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            return loader.load()
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


# --- PART 1: OPTIMIZED INGESTION ENGINE ---

def initialize_retriever(k=2):
    """
    Ingests documents using Multiprocessing and returns a Retriever.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Ensure directories exist
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    # 1. Identify New Files
    indexed_files = set()
    if os.path.exists(INDEXED_FILES_PATH):
        with open(INDEXED_FILES_PATH, 'r') as f:
            indexed_files = set(line.strip() for line in f)

    current_files = set()
    for root, dirs, files in os.walk(DOC_FOLDER):
        for file in files:
            if file.startswith("~$"): continue
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                current_files.add(os.path.join(root, file))

    new_files = list(current_files - indexed_files)

    # 2. Connect to VectorStore
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )

    # 3. Parallel Loading & Ingestion
    if new_files:
        print(f"--- DETECTED {len(new_files)} NEW DOCUMENTS ---")
        docs = []

        # Optimization: Multiprocessing
        # Using ProcessPoolExecutor to load files in parallel
        # max_workers defaults to number of processors
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map file paths to the loader function
            future_to_file = {executor.submit(load_document_by_extension, fp): fp for fp in new_files}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                fp = future_to_file[future]
                try:
                    loaded_docs = future.result()
                    docs.extend(loaded_docs)
                    print(f"   [{i + 1}/{len(new_files)}] Loaded: {os.path.basename(fp)}")
                except Exception as exc:
                    print(f"   [{i + 1}/{len(new_files)}] Failed: {fp} generated {exc}")

        if docs:
            print(f"--- SPLITTING & EMBEDDING {len(docs)} DOCUMENT CHUNKS ---")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            splits = filter_complex_metadata(splits)

            # Batch add to Chroma (Chroma handles batching internally, but we pass all at once)
            vectorstore.add_documents(splits)

            # Update index tracking
            with open(INDEXED_FILES_PATH, 'a') as f:
                for file_path in new_files:
                    f.write(f"{file_path}\n")
            print("--- INGESTION COMPLETE ---")
    else:
        print("--- NO NEW DOCUMENTS TO INGEST ---")

    # Optimization: MMR (Maximal Marginal Relevance) for diversity
    # fetch_k is how many to gather before filtering for diversity
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )


# --- PART 2: MODELS & PROMPTS ---

# Initialize LLMs
thinking_llm = ChatOllama(model=THINKING_OLLAMA_MODEL, temperature=0)
fast_llm = ChatOllama(model=FAST_OLLAMA_MODEL, temperature=0)


# Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured_llm_grader = fast_llm.with_structured_output(GradeDocuments)
grader_system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grader_prompt = ChatPromptTemplate.from_messages(
    [("system", grader_system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
)
grader_chain = grader_prompt | structured_llm_grader

# RAG Generator
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."),
        ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
    ]
)
rag_chain = rag_prompt | thinking_llm | StrOutputParser()

# Rewriter
rewrite_system = "You are a question re-writer that converts an input question to a better version for vector retrieval."
rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", rewrite_system), ("human", "Initial question: \n\n {question} \n Formulate an improved question.")]
)
rewriter_chain = rewrite_prompt | fast_llm | StrOutputParser()


# --- PART 3: GRAPH NODES (OPTIMIZED) ---

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    loop_step: int


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    if GLOBAL_RETRIEVER is None:
        raise ValueError("Retriever not initialized. Run initialize_retriever() first.")

    documents = GLOBAL_RETRIEVER.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE (BATCHED)---")
    question = state["question"]
    documents = state["documents"]

    # Optimization: Batching
    # Prepare inputs for all documents
    batch_inputs = [{"question": question, "document": d.page_content} for d in documents]

    # Run LLM on all docs in parallel/batch
    # This prevents waiting for Doc 1 to finish before starting Doc 2
    scores = grader_chain.batch(batch_inputs)

    filtered_docs = []
    for i, score in enumerate(scores):
        if score.binary_score == "yes":
            print(f"   - Doc {i + 1}: RELEVANT")
            filtered_docs.append(documents[i])
        else:
            print(f"   - Doc {i + 1}: NOT RELEVANT")

    return {"documents": filtered_docs, "question": question}


def generate_rag(state):
    print("---GENERATE RAG---")
    question = state["question"]
    documents = state["documents"]

    formatted_context_list = []
    for doc in documents:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        formatted_context_list.append(f"[Source: {source_name}]\n{doc.page_content}")

    full_context_string = "\n\n---\n\n".join(formatted_context_list)
    #print(f"   - Full context string: {full_context_string}")

    generation = rag_chain.invoke({"context": full_context_string, "question": question})
    return {"generation": generation}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    better_question = rewriter_chain.invoke({"question": question})
    print(f"   - Rewritten: {better_question}")
    return {"documents": documents, "question": better_question, "loop_step": loop_step + 1}


def decide_to_generate(state):
    filtered_documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    if not filtered_documents:
        if loop_step >= 3:
            return "generate"
        return "transform_query"
    return "generate"


# --- PART 4: WORKFLOW BUILD ---

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_rag", generate_rag)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate_rag"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate_rag", END)

app = workflow.compile()


# --- ENTRY POINT ---

def query_documents(user_input: str):
    """Execution wrapper."""
    inputs = {"question": user_input, "loop_step": 0}
    final_generation = "No response."
    try:
        config = {"recursion_limit": 25}
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                if "generation" in value:
                    final_generation = value["generation"]
    except Exception as e:
        return f"Error: {e}"
    return final_generation


print("Initializing RAG System...")
GLOBAL_RETRIEVER = initialize_retriever(k=2)