import os
import sys
import time
import queue
import threading
from typing import List, Optional
import io
import msoffcrypto
import openpyxl

# Concurrency imports
import concurrent.futures
import multiprocessing

# Watchdog for Live Sync
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# --- GLOBAL VARS ---
# Changed from GLOBAL_RETRIEVER to GLOBAL_VECTORSTORE to allow add/delete operations
GLOBAL_VECTORSTORE: Optional[Chroma] = None
INGESTION_QUEUE = queue.Queue()

# --- OCR SETUP ---
try:
    from PIL import Image
    import easyocr

    OCR_AVAILABLE = True
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
    # Safety check for watcher race conditions
    if not os.path.exists(file_path):
        return []

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


# --- LIVE SYNC INFRASTRUCTURE ---

class DocumentEventHandler(FileSystemEventHandler):
    """
    Watchdog Event Handler.
    Pushes events to a thread-safe queue to be processed by the worker.
    """

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"[WATCHER] File Created: {event.src_path}")
            INGESTION_QUEUE.put(("add", event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"[WATCHER] File Modified: {event.src_path}")
            INGESTION_QUEUE.put(("update", event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"[WATCHER] File Deleted: {event.src_path}")
            INGESTION_QUEUE.put(("delete", event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
                print(f"[WATCHER] File Moved (Delete old): {event.src_path}")
                INGESTION_QUEUE.put(("delete", event.src_path))
            if event.dest_path.lower().endswith(SUPPORTED_EXTENSIONS):
                print(f"[WATCHER] File Moved (Add new): {event.dest_path}")
                INGESTION_QUEUE.put(("add", event.dest_path))


def ingestion_worker():
    """
    Background daemon that consumes the queue.
    Handles 'add', 'update', and 'delete' operations on the global VectorStore.
    """
    global GLOBAL_VECTORSTORE

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    while True:
        try:
            # Block until an item is available
            action, file_path = INGESTION_QUEUE.get()

            # Wait briefly to let file writes settle (debounce)
            time.sleep(1.0)

            if GLOBAL_VECTORSTORE is None:
                INGESTION_QUEUE.task_done()
                continue

            print(f"--- SYNC WORKER: Processing {action} for {os.path.basename(file_path)} ---")

            if action == "delete":
                try:
                    GLOBAL_VECTORSTORE.delete(where={"source": file_path})
                    print(f"   - Removed chunks for {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"   - Error deleting {file_path}: {e}")

            elif action in ["add", "update"]:
                # For update, we delete first to avoid duplicates
                if action == "update":
                    try:
                        GLOBAL_VECTORSTORE.delete(where={"source": file_path})
                    except:
                        pass  # Might not exist yet

                # Load and Ingest
                try:
                    docs = load_document_by_extension(file_path)
                    if docs:
                        splits = text_splitter.split_documents(docs)
                        splits = filter_complex_metadata(splits)
                        if splits:
                            GLOBAL_VECTORSTORE.add_documents(splits)
                            print(f"   - Added {len(splits)} chunks for {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"   - Error ingesting {file_path}: {e}")

            INGESTION_QUEUE.task_done()

        except Exception as e:
            print(f"Worker Error: {e}")


# --- PART 1: OPTIMIZED INGESTION ENGINE ---

def initialize_vectorstore():
    """
    Ingests documents using Multiprocessing and returns the VectorStore.
    Also starts the background Watchdog observer.
    """
    global GLOBAL_VECTORSTORE

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Ensure directories exist
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    # 1. Connect to VectorStore
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )
    GLOBAL_VECTORSTORE = vectorstore

    # 2. Identify New Files
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

    # 3. Parallel Loading & Ingestion
    if new_files:
        print(f"--- DETECTED {len(new_files)} NEW DOCUMENTS ---")
        docs = []

        # Optimization: Multiprocessing
        with concurrent.futures.ProcessPoolExecutor() as executor:
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

            vectorstore.add_documents(splits)

            # Update index tracking
            with open(INDEXED_FILES_PATH, 'a') as f:
                for file_path in new_files:
                    f.write(f"{file_path}\n")
            print("--- INGESTION COMPLETE ---")
    else:
        print("--- NO NEW DOCUMENTS TO INGEST ---")

    # 4. Start Background Sync (Watchdog)
    print("--- STARTING LIVE DOC WATCHER ---")

    # Start Worker Thread
    worker_thread = threading.Thread(target=ingestion_worker, daemon=True)
    worker_thread.start()

    # Start Observer
    event_handler = DocumentEventHandler()
    observer = Observer()
    observer.schedule(event_handler, DOC_FOLDER, recursive=True)
    observer.start()

    return vectorstore


def get_retriever(k=2):
    """Returns a retriever interface from the current global vectorstore."""
    global GLOBAL_VECTORSTORE
    if GLOBAL_VECTORSTORE is None:
        initialize_vectorstore()

    return GLOBAL_VECTORSTORE.as_retriever(
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

rag_system_prompt = """You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
The context is formatted as: 
[Source: filename | Location: page or line number]
Content...

When answering:
1. Cite your sources in the text using the format [Source Name, Page/Line].
2. If the context does not contain the answer, say "I don't know."
"""

# RAG Generator
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
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

    # Always fetch a fresh retriever instance to ensure it sees latest DB updates
    retriever = get_retriever(k=2)

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE (BATCHED)---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "question": question}

    # Optimization: Batching
    batch_inputs = [{"question": question, "document": d.page_content} for d in documents]

    # Run LLM on all docs in parallel/batch
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
        # 1. Extract Source Name
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))

        # 2. Extract Location (Page for PDF, Start Index/Line approx for Text)
        location = ""
        if "page" in doc.metadata:
            # PyPDFLoader is 0-indexed
            location = f"Page {doc.metadata['page'] + 1}"
        elif "start_index" in doc.metadata:
            location = f"Char Index {doc.metadata['start_index']}"
        else:
            location = "Unknown Location"

        # 3. Format the context for the LLM
        formatted_entry = f"[Source: {source_name} | Location: {location}]\n{doc.page_content}"
        formatted_context_list.append(formatted_entry)

    full_context_string = "\n\n---\n\n".join(formatted_context_list)

    print(f"   - Streaming response from {THINKING_OLLAMA_MODEL}...")
    generation = ""

    # Stream the answer
    for chunk in rag_chain.stream({"context": full_context_string, "question": question}):
        print(chunk, end="", flush=True)
        generation += chunk

    print("\n")

    # Return generation AND pass the documents through
    return {"generation": generation, "documents": documents}


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
    """Execution wrapper returning Answer + Sources."""
    inputs = {"question": user_input, "loop_step": 0}

    final_state = None

    # Run the graph
    config = {"recursion_limit": 25}
    # Ensure DB is initialized before first query
    if GLOBAL_VECTORSTORE is None:
        initialize_vectorstore()

    try:
        # We iterate through the stream to print progress, but we need the final state
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                final_state = value
    except Exception as e:
        return {"error": str(e)}

    # Extract final generation
    final_answer = final_state.get("generation", "No answer generated.")
    used_docs = final_state.get("documents", [])

    # Format the sources list programmatically
    sources_data = []
    for doc in used_docs:
        meta = doc.metadata
        filename = os.path.basename(meta.get("source", "Unknown"))

        # logic to determine location
        location = "N/A"
        if "page" in meta:
            location = f"Page {meta['page'] + 1}"
        elif "start_index" in meta:
            location = f"Start Char {meta['start_index']}"

        sources_data.append({
            "file": filename,
            "location": location,
            "snippet": doc.page_content[:50] + "..."  # First 50 chars as preview
        })

    # Return a dictionary containing both Answer and Structured Sources
    return {
        "answer": final_answer,
        "citations": sources_data
    }

with open("document_engine_diagram.png", "wb") as binary_file:
    binary_file.write(app.get_graph().draw_mermaid_png())