import atexit
import os
import time
import queue
import threading
import json
import logging
from typing import List, Optional
import io
# msoffcrypto + openpyxl are pulled in by UnstructuredExcelLoader at runtime;
# importing them here surfaces missing-dependency errors at startup instead of
# the first xlsx ingest.
import msoffcrypto  # noqa: F401  — capability probe for encrypted xlsx support
import openpyxl  # noqa: F401  — capability probe for xlsx support

# Concurrency imports
import concurrent.futures

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
    OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT, THINKING_OLLAMA_MODEL, EMBEDDING_MODEL
from observability import init_logging, METRICS

# Define supported file extensions
SUPPORTED_EXTENSIONS = ('.docx', '.pdf', '.xlsx', '.csv', '.txt', '.md')

# --- GLOBAL VARS ---
# Changed from GLOBAL_RETRIEVER to GLOBAL_VECTORSTORE to allow add/delete operations
GLOBAL_VECTORSTORE: Optional[Chroma] = None
# Guards reads/writes of GLOBAL_VECTORSTORE. The watchdog worker thread and the
# main thread both touch it; without a lock, an in-flight initialize_vectorstore()
# could race with an ingestion event and leave the reference dangling.
VECTORSTORE_LOCK = threading.RLock()
INGESTION_QUEUE = queue.Queue()
MANIFEST_LOCK = threading.Lock()

# Sentinel pushed onto INGESTION_QUEUE during shutdown to unblock the worker.
_SHUTDOWN_SENTINEL = ("__shutdown__", None)
_OBSERVER = None
_WORKER_THREAD: Optional[threading.Thread] = None

init_logging()
logger = logging.getLogger(__name__)

# --- OCR SETUP ---
try:
    from PIL import Image
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. OCR fallback unavailable.")

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Cannot render PDF pages for OCR.")


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
        METRICS.record_error("document_engine.pdf", e)
        logger.warning("Error processing PDF %s: %s", file_path, e)
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
        METRICS.record_error("document_engine.load", e)
        logger.warning("Error loading %s: %s", file_path, e)
        return []


# --- LIVE SYNC INFRASTRUCTURE ---

class DocumentEventHandler(FileSystemEventHandler):
    """
    Watchdog Event Handler.
    Pushes events to a thread-safe queue to be processed by the worker.
    """

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.info("[WATCHER] File Created: %s", event.src_path)
            INGESTION_QUEUE.put(("add", event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.info("[WATCHER] File Modified: %s", event.src_path)
            INGESTION_QUEUE.put(("update", event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.info("[WATCHER] File Deleted: %s", event.src_path)
            INGESTION_QUEUE.put(("delete", event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
                logger.info("[WATCHER] File Moved (Delete old): %s", event.src_path)
                INGESTION_QUEUE.put(("delete", event.src_path))
            if event.dest_path.lower().endswith(SUPPORTED_EXTENSIONS):
                logger.info("[WATCHER] File Moved (Add new): %s", event.dest_path)
                INGESTION_QUEUE.put(("add", event.dest_path))


def _get_file_signature(file_path: str) -> dict:
    return {
        "mtime": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
    }


def _load_index_manifest() -> dict:
    if not os.path.exists(INDEXED_FILES_PATH):
        return {}
    try:
        with open(INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        if raw.startswith("{"):
            return json.loads(raw)
        # Backward compatibility with legacy newline list
        return {line.strip(): {} for line in raw.splitlines() if line.strip()}
    except Exception as e:
        METRICS.record_error("document_engine.manifest_read", e)
        logger.warning("Error reading index manifest: %s", e)
        return {}


def _write_index_manifest(manifest: dict) -> None:
    try:
        with open(INDEXED_FILES_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        METRICS.record_error("document_engine.manifest_write", e)
        logger.warning("Error writing index manifest: %s", e)


def _update_manifest(action: str, file_path: str) -> None:
    with MANIFEST_LOCK:
        manifest = _load_index_manifest()
        if action == "delete":
            manifest.pop(file_path, None)
            _write_index_manifest(manifest)
            return
        if os.path.exists(file_path):
            manifest[file_path] = _get_file_signature(file_path)
            _write_index_manifest(manifest)


def _process_one_ingestion(vectorstore, action: str, file_path: str, text_splitter) -> None:
    """Apply a single (action, file_path) to the vectorstore.

    Extracted from ingestion_worker so the coalescing logic above stays
    readable and testable.
    """
    logger.info("--- SYNC WORKER: Processing %s for %s ---", action, os.path.basename(file_path))

    if action == "delete":
        try:
            vectorstore.delete(where={"source": file_path})
            _update_manifest("delete", file_path)
            logger.info("   - Removed chunks for %s", os.path.basename(file_path))
        except Exception as e:
            METRICS.record_error("document_engine.delete", e)
            logger.warning("   - Error deleting %s: %s", file_path, e)
        return

    if action in ("add", "update"):
        # For update, delete first so we don't accumulate duplicate chunks.
        # A miss is expected on first ingest of a file, so debug-level only.
        if action == "update":
            try:
                vectorstore.delete(where={"source": file_path})
            except Exception as e:
                logger.debug("   - Pre-update delete miss for %s: %s", file_path, e)
        try:
            docs = load_document_by_extension(file_path)
            if docs:
                splits = text_splitter.split_documents(docs)
                splits = filter_complex_metadata(splits)
                if splits:
                    vectorstore.add_documents(splits)
                    _update_manifest("add", file_path)
                    logger.info("   - Added %d chunks for %s", len(splits), os.path.basename(file_path))
        except Exception as e:
            METRICS.record_error("document_engine.ingest", e)
            logger.warning("   - Error ingesting %s: %s", file_path, e)


def ingestion_worker():
    """
    Background daemon that consumes the queue.
    Handles 'add', 'update', and 'delete' operations on the global VectorStore.
    Exits cleanly when it receives _SHUTDOWN_SENTINEL.

    Phase 14: after pulling the first event, the worker waits ~1s for the
    burst to settle, then drains every queued event and coalesces by
    file_path (last-write-wins). A LibreOffice save that emits 4 'modified'
    events for one file now produces a single ingest instead of four.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    while True:
        try:
            # Block until at least one event arrives.
            first_item = INGESTION_QUEUE.get()

            # Brief debounce — let burst events arrive.
            time.sleep(1.0)

            # Drain everything currently in the queue. Non-blocking — we just
            # take whatever's already there at this moment.
            items = [first_item]
            while True:
                try:
                    items.append(INGESTION_QUEUE.get_nowait())
                except queue.Empty:
                    break

            shutdown = any(i == _SHUTDOWN_SENTINEL for i in items)
            real_items = [i for i in items if i != _SHUTDOWN_SENTINEL]

            # Coalesce by file_path. Last action wins per path. The previous
            # implementation processed every event in arrival order; this is
            # equivalent for the common "save the same file 4 times" case and
            # dramatically faster.
            pending: dict[str, str] = {}
            for action, file_path in real_items:
                pending[file_path] = action

            coalesced = len(real_items) - len(pending)
            if coalesced > 0:
                METRICS.increment("tool.ingest_coalesced", value=coalesced)
                logger.debug("Coalesced %d duplicate ingest event(s) across %d path(s)",
                             coalesced, len(pending))

            with VECTORSTORE_LOCK:
                vectorstore = GLOBAL_VECTORSTORE

            if vectorstore is not None:
                for file_path, action in pending.items():
                    _process_one_ingestion(vectorstore, action, file_path, text_splitter)

            # Mark every consumed item done so queue.join() works.
            for _ in items:
                INGESTION_QUEUE.task_done()

            if shutdown:
                logger.info("Ingestion worker received shutdown signal; exiting")
                return

        except Exception as e:
            METRICS.record_error("document_engine.worker", e)
            logger.warning("Worker error: %s", e)


# --- PART 1: OPTIMIZED INGESTION ENGINE ---

def initialize_vectorstore():
    """
    Ingests documents using Multiprocessing and returns the VectorStore.
    Also starts the background Watchdog observer.
    """
    global GLOBAL_VECTORSTORE, _OBSERVER, _WORKER_THREAD

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
    with VECTORSTORE_LOCK:
        GLOBAL_VECTORSTORE = vectorstore

    # 2. Identify New Files
    manifest = _load_index_manifest()
    indexed_files = set(manifest.keys())
    current_files = {}
    for root, dirs, files in os.walk(DOC_FOLDER):
        for file in files:
            if file.startswith("~$"): continue
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                path = os.path.join(root, file)
                current_files[path] = _get_file_signature(path)

    current_paths = set(current_files.keys())
    deleted_files = list(indexed_files - current_paths)
    candidate_files = list(current_paths)

    new_files = []
    updated_files = []
    for file_path in candidate_files:
        if file_path not in manifest:
            new_files.append(file_path)
            continue
        if manifest.get(file_path) != current_files[file_path]:
            updated_files.append(file_path)

    # 3. Parallel Loading & Ingestion
    if deleted_files:
        logger.info("--- DETECTED %d DELETED DOCUMENTS ---", len(deleted_files))
        for file_path in deleted_files:
            try:
                vectorstore.delete(where={"source": file_path})
                _update_manifest("delete", file_path)
                logger.info("   - Removed chunks for %s", os.path.basename(file_path))
            except Exception as e:
                METRICS.record_error("document_engine.delete", e)
                logger.warning("   - Error deleting %s: %s", file_path, e)

    files_to_ingest = new_files + updated_files

    if files_to_ingest:
        if new_files:
            logger.info("--- DETECTED %d NEW DOCUMENTS ---", len(new_files))
        if updated_files:
            logger.info("--- DETECTED %d UPDATED DOCUMENTS ---", len(updated_files))

        for file_path in updated_files:
            try:
                vectorstore.delete(where={"source": file_path})
            except Exception as e:
                # Miss is expected if the file was never indexed before.
                logger.debug("Pre-update delete miss for %s: %s", file_path, e)

        docs = []

        # Optimization: Multiprocessing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(load_document_by_extension, fp): fp for fp in files_to_ingest}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
                fp = future_to_file[future]
                try:
                    loaded_docs = future.result()
                    docs.extend(loaded_docs)
                    logger.info("   [%d/%d] Loaded: %s", i + 1, len(files_to_ingest), os.path.basename(fp))
                except Exception as exc:
                    METRICS.record_error("document_engine.load", exc)
                    logger.warning("   [%d/%d] Failed: %s generated %s", i + 1, len(files_to_ingest), fp, exc)

        if docs:
            logger.info("--- SPLITTING & EMBEDDING %d DOCUMENT CHUNKS ---", len(docs))
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            splits = filter_complex_metadata(splits)

            vectorstore.add_documents(splits)

            # Update index tracking
            for file_path in files_to_ingest:
                _update_manifest("add", file_path)
            logger.info("--- INGESTION COMPLETE ---")
    else:
        logger.info("--- NO DOCUMENT CHANGES TO INGEST ---")

    # 4. Start Background Sync (Watchdog)
    logger.info("--- STARTING LIVE DOC WATCHER ---")

    # Start Worker Thread
    _WORKER_THREAD = threading.Thread(target=ingestion_worker, name="ingestion-worker", daemon=True)
    _WORKER_THREAD.start()

    # Start Observer
    event_handler = DocumentEventHandler()
    _OBSERVER = Observer()
    _OBSERVER.schedule(event_handler, DOC_FOLDER, recursive=True)
    _OBSERVER.start()

    return vectorstore


def shutdown(timeout: float = 5.0) -> None:
    """Stop the watchdog observer and ingestion worker cleanly.

    Safe to call multiple times. Intended for bot shutdown or atexit.
    """
    global _OBSERVER, _WORKER_THREAD

    if _OBSERVER is not None:
        try:
            _OBSERVER.stop()
            _OBSERVER.join(timeout=timeout)
        except Exception as e:
            logger.warning("Error stopping watchdog observer: %s", e)
        _OBSERVER = None

    if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
        INGESTION_QUEUE.put(_SHUTDOWN_SENTINEL)
        _WORKER_THREAD.join(timeout=timeout)
        if _WORKER_THREAD.is_alive():
            logger.warning("Ingestion worker did not exit within %.1fs", timeout)
    _WORKER_THREAD = None


def get_retriever(k=2):
    """Returns a retriever interface from the current global vectorstore."""
    with VECTORSTORE_LOCK:
        vectorstore = GLOBAL_VECTORSTORE
    if vectorstore is None:
        initialize_vectorstore()
        with VECTORSTORE_LOCK:
            vectorstore = GLOBAL_VECTORSTORE

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )


# --- PART 2: MODELS & PROMPTS ---

# Initialize LLMs
thinking_llm = ChatOllama(
    model=THINKING_OLLAMA_MODEL, temperature=0,
    keep_alive=OLLAMA_KEEP_ALIVE,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)
fast_llm = ChatOllama(
    model=FAST_OLLAMA_MODEL, temperature=0,
    keep_alive=OLLAMA_KEEP_ALIVE,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)


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

def query_documents(user_input: str, include_sources: bool = False):
    """Execution wrapper returning Answer (and optional structured sources)."""
    inputs = {"question": user_input, "loop_step": 0}

    final_state = None

    # Run the graph
    config = {"recursion_limit": 25}
    # Ensure DB is initialized before first query
    with VECTORSTORE_LOCK:
        needs_init = GLOBAL_VECTORSTORE is None
    if needs_init:
        initialize_vectorstore()

    try:
        # We iterate through the stream to print progress, but we need the final state
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                final_state = value
    except Exception as e:
        METRICS.record_error("document_engine.query", e)
        logger.warning("Query error: %s", e)
        return f"Error: {e}" if not include_sources else {"error": str(e)}

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
    if include_sources:
        return {
            "answer": final_answer,
            "citations": sources_data
        }
    return final_answer

# FRITZ_WRITE_DIAGRAMS=0 (set by tests/conftest.py) keeps the writer from
# dirtying the tracked PNG on every test run.
if os.environ.get("FRITZ_WRITE_DIAGRAMS", "1") != "0":
    try:
        with open("document_engine_diagram.png", "wb") as binary_file:
            binary_file.write(app.get_graph().draw_mermaid_png())
    except Exception as _diagram_err:
        # draw_mermaid_png hits a remote service; we don't want import to fail
        # in offline or sandboxed environments.
        logger.debug("Could not write document_engine diagram (non-fatal): %s", _diagram_err)

# Ensure watchdog observer and ingestion worker exit cleanly on interpreter shutdown.
atexit.register(shutdown)
