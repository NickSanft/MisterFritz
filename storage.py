import json
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_ollama import OllamaEmbeddings
from typing_extensions import Literal

from fritz_utils import CHROMA_DB_PATH, EMBEDDING_MODEL

_embeddings = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


class SQLiteStore(BaseStore[str, Union[str, bytes]]):
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    namespace TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (namespace, key)
                )
            """)
            conn.commit()

    def _execute_query(self, query: str, params: Tuple = ()):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        result = self._execute_query("SELECT value FROM store WHERE key = ?", (key,))
        return json.loads(result[0][0]) if result else None

    def mget(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        placeholders = ','.join(['?'] * len(keys))
        query = f"SELECT key, value FROM store WHERE key IN ({placeholders})"
        result = self._execute_query(query, tuple(keys))
        result_dict = {key: json.loads(value) for key, value in result}
        return [result_dict.get(key) for key in keys]

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any],
            index: Literal[False] | List[str] | None = None) -> None:
        namespace_str = "/".join(namespace)
        value_str = json.dumps(value)
        self.mset([(namespace_str, key, value_str)])

    def mset(self, key_value_pairs: List[Tuple[str, str, str]]) -> None:
        query = "REPLACE INTO store (namespace, key, value) VALUES (?, ?, ?)"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(query, key_value_pairs)
            conn.commit()

    def delete(self, key: str) -> None:
        self.mdelete([key])

    def mdelete(self, keys: List[str]) -> None:
        placeholders = ','.join(['?'] * len(keys))
        query = f"DELETE FROM store WHERE key IN ({placeholders})"
        self._execute_query(query, tuple(keys))

    def yield_keys(self, prefix: Optional[str] = "") -> Iterator[str]:
        query = "SELECT key FROM store WHERE key LIKE ?"
        for row in self._execute_query(query, (f"{prefix}%",)):
            yield row[0]

    def search(self, namespace: Tuple[str, ...], limit: int) -> List[Tuple[str, Dict[str, Any]]]:
        namespace_str = "/".join(namespace)
        sql_query = "SELECT key, value FROM store WHERE namespace = ? LIMIT ?"
        results = self._execute_query(sql_query, (namespace_str, limit))
        return [(key, json.loads(value)) for key, value in results]


class ChromaStore(BaseStore[str, Union[str, bytes]]):
    """Vector-store backed KV store using ChromaDB with semantic search support."""

    def __init__(
            self,
            collection_name: str = "langchain_store",
            persist_directory: Optional[str] = CHROMA_DB_PATH,
    ):
        embeddings = _get_embeddings()
        self.embedding_function = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        result = self.vectorstore.get(ids=[key], include=["metadatas"])
        if result and result['metadatas']:
            return result['metadatas'][0]
        return None

    def mget(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        if not keys:
            return []
        result = self.vectorstore.get(ids=keys, include=["metadatas"])
        found_map = {}
        if result and result['ids']:
            for id_str, metadata in zip(result['ids'], result['metadatas']):
                found_map[id_str] = metadata
        return [found_map.get(key) for key in keys]

    def put(self, namespace: Tuple[str, ...], key: str, value: Dict[str, Any],
            index: Literal[False] | List[str] | None = None) -> None:
        namespace_str = "/".join(namespace)
        self.mset([(namespace_str, key, value)])

    def mset(self, key_value_pairs: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        documents = []
        ids = []
        for namespace_str, key, value in key_value_pairs:
            metadata = value.copy()
            metadata["namespace"] = namespace_str
            metadata["original_key"] = key
            if "page_content" in value:
                content = str(value["page_content"])
            elif "text" in value:
                content = str(value["text"])
            elif "content" in value:
                content = str(value["content"])
            else:
                content = json.dumps(value)
            documents.append(Document(page_content=content, metadata=metadata))
            ids.append(key)
        if documents:
            self.vectorstore.add_documents(documents=documents, ids=ids)

    def delete(self, key: str) -> None:
        self.mdelete([key])

    def mdelete(self, keys: List[str]) -> None:
        if keys:
            self.vectorstore.delete(ids=keys)

    def yield_keys(self, prefix: Optional[str] = "") -> Iterator[str]:
        results = self.vectorstore.get(include=[])
        all_ids = results.get("ids", [])
        for doc_id in all_ids:
            if prefix is None or doc_id.startswith(prefix):
                yield doc_id

    def search(self, query: str, namespace: Tuple[str, ...], limit: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
        namespace_str = "/".join(namespace)
        results = self.vectorstore.similarity_search(
            query,
            k=limit,
            filter={"namespace": namespace_str},
        )
        return [(doc.metadata.get("original_key"), doc.metadata) for doc in results]
