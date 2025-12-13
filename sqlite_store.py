import sqlite3
import json

from langchain_core.stores import BaseStore
from typing import List, Tuple, Optional, Union, Iterator, Dict, Any
from typing_extensions import Literal


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

    def search(self, namespace: Tuple[str, ...], limit: int) -> List[
        Tuple[str, Dict[str, Any]]]:
        namespace_str = "/".join(namespace)
        sql_query = "SELECT key, value FROM store WHERE namespace = ? LIMIT ?"
        print(sql_query)
        results = self._execute_query(sql_query, (namespace_str, limit))

        return [(key, json.loads(value)) for key, value in results]
