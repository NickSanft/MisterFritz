import json
from typing import List, Tuple, Optional, Union, Iterator, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_ollama import OllamaEmbeddings
from typing_extensions import Literal

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

class ChromaStore(BaseStore[str, Union[str, bytes]]):
    """
    A vector-store backed implementation of a Key-Value store using ChromaDB.
    Allows for semantic searching in addition to standard key-value retrieval.
    """

    def __init__(
            self,
            collection_name: str = "langchain_store",
            persist_directory: Optional[str] = 'chroma_store'
    ):
        """
        Initialize the ChromaStore.5

        Args:
            collection_name: The name of the Chroma collection.
            persist_directory: Path to store the database on disk. If None, uses in-memory.
        """
        self.embedding_function = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize the Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a value by key."""
        # Chroma get returns a dictionary with lists of results
        result = self.vectorstore.get(ids=[key], include=["metadatas"])

        if result and result['metadatas']:
            # Return the first match's metadata (reconstructed to original dict)
            # We strip internal metadata keys if necessary, but here we return full metadata
            return result['metadatas'][0]
        return None

    def mget(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Retrieve multiple values by keys."""
        if not keys:
            return []

        result = self.vectorstore.get(ids=keys, include=["metadatas"])

        # Create a map for O(1) lookup to ensure order matches input 'keys'
        found_map = {}
        if result and result['ids']:
            for id_str, metadata in zip(result['ids'], result['metadatas']):
                found_map[id_str] = metadata

        return [found_map.get(key) for key in keys]

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any],
            index: Literal[False] | List[str] | None = None) -> None:
        """
        Insert or update a single key-value pair.

        The 'value' dict is stored in metadata.
        For semantic search, we need a text representation (page_content).
        This implementation looks for a 'text', 'content', or 'page_content' key in 'value'.
        If none are found, it dumps the entire JSON as the content to embed.
        """
        namespace_str = "/".join(namespace)
        self.mset([(namespace_str, key, value)])

    def mset(self, key_value_pairs: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Insert or update multiple key-value pairs.

        Args:
            key_value_pairs: List of (namespace_str, key, value_dict)
        """
        documents = []
        ids = []

        for namespace_str, key, value in key_value_pairs:
            # 1. Prepare Metadata
            # We enforce the namespace and original key into metadata for filtering later
            metadata = value.copy()
            metadata["namespace"] = namespace_str
            metadata["original_key"] = key

            # 2. Determine Content for Embedding
            # Prefer specific keys for semantic meaning, fallback to JSON dump
            if "page_content" in value:
                content = str(value["page_content"])
            elif "text" in value:
                content = str(value["text"])
            elif "content" in value:
                content = str(value["content"])
            else:
                content = json.dumps(value)

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            ids.append(key)

        # Add to Chroma (upsert logic handles replacements based on ID)
        if documents:
            self.vectorstore.add_documents(documents=documents, ids=ids)

    def delete(self, key: str) -> None:
        """Delete a specific key."""
        self.mdelete([key])

    def mdelete(self, keys: List[str]) -> None:
        """Delete multiple keys."""
        if keys:
            self.vectorstore.delete(ids=keys)

    def yield_keys(self, prefix: Optional[str] = "") -> Iterator[str]:
        """
        Yield keys, optionally filtering by a prefix.
        Note: Vector stores are not optimized for prefix scans.
        This fetches all IDs and filters in memory.
        """
        # Get all IDs from the collection
        results = self.vectorstore.get(include=[])
        all_ids = results.get("ids", [])

        for doc_id in all_ids:
            if prefix is None or doc_id.startswith(prefix):
                yield doc_id

    def search(self, query: str, namespace: Tuple[str, ...], limit: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Perform a semantic similarity search.

        Args:
            query: The text to search for semantically.
            namespace: The namespace to filter search results by.
            limit: Number of results to return.

        Returns:
            List of (key, value_dict) tuples.
        """
        namespace_str = "/".join(namespace)

        # Perform similarity search with metadata filter for the namespace
        results = self.vectorstore.similarity_search(
            query,
            k=limit,
            filter={"namespace": namespace_str}
        )

        output = []
        for doc in results:
            # We use the ID stored in Chroma (which we set as the key)
            # However, doc object usually doesn't expose ID directly in simple search
            # unless we use similarity_search_with_score or custom retrieval.
            # Fortunately, we stored 'original_key' in metadata.

            key = doc.metadata.get("original_key")
            output.append((key, doc.metadata))

        return output