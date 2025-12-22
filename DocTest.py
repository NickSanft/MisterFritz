import os
import sys
from typing import List, Literal

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import END, StateGraph, START

# --- CONFIGURATION ---
# Ensure you have your OpenAI API Key set in your environment
# os.environ["OPENAI_API_KEY"] = "sk-..."

DOCS_FOLDER = "./input"  # Folder containing your .docx files
DB_PATH = "./chroma_store"  # Where the vector DB will be saved
COLLECTION_NAME = "word_docs_rag"
INDEXED_FILES_PATH = os.path.join(DB_PATH, "indexed_files.txt")


# --- PART 1: INGESTION ENGINE ---
def get_vectorstore_retriever():
    """
    Checks if a local vector store exists. If not, ingests documents from DOCS_FOLDER.
    If vector store exists, checks for new documents and adds them.
    Returns a retriever object.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Check if DB exists
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("--- LOADING EXISTING VECTOR STORE ---")
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # Check for new documents
        indexed_files = set()
        if os.path.exists(INDEXED_FILES_PATH):
            with open(INDEXED_FILES_PATH, 'r') as f:
                indexed_files = set(line.strip() for line in f)

        # Find all current .docx files
        current_files = set()
        if os.path.exists(DOCS_FOLDER):
            for root, dirs, files in os.walk(DOCS_FOLDER):
                for file in files:
                    if file.endswith('.docx'):
                        current_files.add(os.path.join(root, file))

        new_files = current_files - indexed_files

        if new_files:
            print(f"--- FOUND {len(new_files)} NEW DOCUMENTS ---")
            # Load and process only new files
            docs = []
            for file_path in new_files:
                print(f"   - Loading: {file_path}")
                loader = UnstructuredWordDocumentLoader(file_path)
                docs.extend(loader.load())

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    add_start_index=True
                )
                splits = text_splitter.split_documents(docs)
                vectorstore.add_documents(splits)

                # Update indexed files list
                with open(INDEXED_FILES_PATH, 'a') as f:
                    for file_path in new_files:
                        f.write(f"{file_path}\n")

                print("--- NEW DOCUMENTS ADDED ---")
        else:
            print("--- NO NEW DOCUMENTS FOUND ---")
    else:
        print("--- CREATING NEW VECTOR STORE FROM DOCUMENTS ---")
        if not os.path.exists(DOCS_FOLDER):
            os.makedirs(DOCS_FOLDER)
            print(f"Created folder {DOCS_FOLDER}. Please add .docx files and restart.")
            sys.exit()

        # 1. Load Word Documents
        # UnstructuredWordDocumentLoader is robust for unformatted text
        loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()

        if not docs:
            print("No documents found. Please add .docx files to the folder.")
            sys.exit()

        # 2. Split Text
        # Large chunks + overlap help maintain context in messy docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        # 3. Index
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_PATH
        )
        print("--- INGESTION COMPLETE ---")

        # Track indexed files
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
        with open(INDEXED_FILES_PATH, 'w') as f:
            for root, dirs, files in os.walk(DOCS_FOLDER):
                for file in files:
                    if file.endswith('.docx'):
                        f.write(f"{os.path.join(root, file)}\n")

    return vectorstore.as_retriever()


# --- PART 2: STATE DEFINITION ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    loop_step: int  # Tracks retry attempts


# --- PART 3: PROMPTS & MODELS ---
llm = ChatOllama(model="gpt-oss", temperature=0)


# B. Document Grader Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)
grader_system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grader_prompt = ChatPromptTemplate.from_messages(
    [("system", grader_system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
)
grader_chain = grader_prompt | structured_llm_grader

# C. RAG Generator
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."),
        ("human", "Question: {question} \n\n Context: {context} \n\n Answer:"),
    ]
)
rag_chain = rag_prompt | llm | StrOutputParser()

# D. Query Rewriter
rewrite_system = """You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. 
Look at the initial and formulate an improved question."""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", rewrite_system),
     ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")]
)
rewriter_chain = rewrite_prompt | llm | StrOutputParser()


# --- PART 4: NODES ---

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    # We initialize the retriever here to avoid pickling issues if passed in state
    retriever = get_vectorstore_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("   - Grade: RELEVANT")
            filtered_docs.append(d)
        else:
            print("   - Grade: NOT RELEVANT")

    return {"documents": filtered_docs, "question": question}


def generate_rag(state):
    print("---GENERATE RAG---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
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
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    if not filtered_documents:
        # If we have looped too many times, just force generation (or end)
        if loop_step >= 3:
            print("   - Max retries reached. Forcing generation.")
            return "generate"
        # Otherwise, rewrite query
        return "transform_query"

    # We have relevant docs
    return "generate"


# --- PART 6: BUILD GRAPH ---
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_rag", generate_rag)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START,"retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate_rag",
    },
)

workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate_rag", END)

# Compile
app = workflow.compile()

def ask_question(user_input):
    # Run the graph
    inputs = {
        "question": user_input,
        "loop_step": 0
    }
    final_generation = ""
    for output in app.stream(inputs):
        print("TYPE: " + str(type(output)))
        for key, value in output.items():
            # print(f"Finished running: {key}")
            if "generation" in value:
                final_generation = value["generation"]

    print(f"Agent: {final_generation}")
    return app.invoke(inputs)["generation"]

if __name__ == '__main__':
    answer = ask_question("Who is Senialis?")
    print("FINAL ANSWER: ", answer)
