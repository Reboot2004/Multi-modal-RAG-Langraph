import os
import sqlite3
import time
from typing import Dict, List

from config.settings import LEXICAL_DB_PATH
from pipeline_logger import get_logger


logger = get_logger("lexical_store")


class LexicalStore:
    def __init__(self):
        self.db_path = LEXICAL_DB_PATH
        self._ensure_parent_dir()
        self._initialize()
        logger.info("LexicalStore initialized | db=%s", self.db_path)

    def _ensure_parent_dir(self):
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _is_malformed_error(self, ex: Exception) -> bool:
        text = str(ex).lower()
        return "malformed" in text or "database disk image" in text

    def _recover_malformed_db(self, reason: str):
        backup_or_remove_failed = False

        if os.path.exists(self.db_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.corrupt_{timestamp}.bak"

            try:
                os.replace(self.db_path, backup_path)
                logger.warning(
                    "Detected malformed lexical DB; moved to backup | reason=%s | backup=%s",
                    reason,
                    backup_path,
                )
            except Exception as backup_ex:
                logger.warning(
                    "Failed to backup malformed lexical DB; removing file | reason=%s | error=%s",
                    reason,
                    backup_ex,
                )
                try:
                    os.remove(self.db_path)
                except Exception as remove_ex:
                    backup_or_remove_failed = True
                    logger.warning(
                        "Failed to remove malformed lexical DB | path=%s | error=%s",
                        self.db_path,
                        remove_ex,
                    )

        # Remove SQLite sidecar files that may hold stale/corrupt state.
        for suffix in ("-wal", "-shm"):
            sidecar = f"{self.db_path}{suffix}"
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                except Exception:
                    pass

        if backup_or_remove_failed:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fresh_path = f"{self.db_path}.recovered_{timestamp}.db"
            logger.warning(
                "Switching lexical store to fresh DB due to file lock | old=%s | new=%s",
                self.db_path,
                fresh_path,
            )
            self.db_path = fresh_path

        self._initialize()

    def _initialize(self):
        for attempt in range(2):
            conn = None
            try:
                conn = self._connect()
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id INTEGER PRIMARY KEY,
                        source TEXT,
                        page INTEGER,
                        language TEXT,
                        text TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, content='chunks', content_rowid='chunk_id')
                    """
                )
                conn.commit()
                return
            except sqlite3.DatabaseError as ex:
                if attempt == 0 and self._is_malformed_error(ex):
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    self._recover_malformed_db(reason=str(ex))
                    continue
                raise
            finally:
                if conn is not None:
                    conn.close()

    def upsert_chunks(self, chunks: List[Dict]):
        if not chunks:
            return

        for attempt in range(2):
            conn = None
            try:
                conn = self._connect()
                cur = conn.cursor()
                for item in chunks:
                    chunk_id = item.get("chunk_id")
                    if not isinstance(chunk_id, int):
                        continue

                    source = str(item.get("source", ""))
                    page = int(item.get("page", 0) or 0)
                    language = str(item.get("language", ""))
                    text = str(item.get("text", ""))

                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_id, source, page, language, text)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            source=excluded.source,
                            page=excluded.page,
                            language=excluded.language,
                            text=excluded.text
                        """,
                        (chunk_id, source, page, language, text),
                    )

                    cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
                    cur.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, text))

                conn.commit()
                return
            except sqlite3.DatabaseError as ex:
                if attempt == 0 and self._is_malformed_error(ex):
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    self._recover_malformed_db(reason=str(ex))
                    continue
                raise
            finally:
                if conn is not None:
                    conn.close()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not query or not query.strip():
            return []

        rows = []
        for attempt in range(2):
            conn = None
            try:
                conn = self._connect()
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT c.chunk_id, c.source, c.page, c.language, c.text, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, int(top_k)),
                )
                rows = cur.fetchall()
                break
            except sqlite3.OperationalError as ex:
                # FTS query parser can reject malformed queries; fallback to phrase query.
                logger.warning("Lexical search fallback due to query parser error | error=%s", ex)
                cleaned_query = query.replace('"', "")
                safe_query = f'"{cleaned_query}"'
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT c.chunk_id, c.source, c.page, c.language, c.text, bm25(chunks_fts) AS rank
                        FROM chunks_fts
                        JOIN chunks c ON c.chunk_id = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (safe_query, int(top_k)),
                    )
                    rows = cur.fetchall()
                except sqlite3.OperationalError:
                    like_query = f"%{cleaned_query}%"
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT chunk_id, source, page, language, text, 0.0 AS rank
                        FROM chunks
                        WHERE text LIKE ?
                        LIMIT ?
                        """,
                        (like_query, int(top_k)),
                    )
                    rows = cur.fetchall()
                break
            except sqlite3.DatabaseError as ex:
                if attempt == 0 and self._is_malformed_error(ex):
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    self._recover_malformed_db(reason=str(ex))
                    continue
                raise
            finally:
                if conn is not None:
                    conn.close()

        results = []
        for position, row in enumerate(rows, start=1):
            chunk_id, source, page, language, text, rank = row
            lexical_score = 1.0 / (60 + position)
            results.append(
                {
                    "score": float(lexical_score),
                    "rank": float(rank if rank is not None else 0.0),
                    "text": text,
                    "metadata": {
                        "chunk_id": int(chunk_id),
                        "source": source,
                        "page": int(page),
                        "language": language,
                        "text": text,
                    },
                }
            )

        logger.debug("Lexical search completed | query_chars=%d | returned=%d", len(query), len(results))
        return results

    def clear(self):
        for attempt in range(2):
            conn = None
            try:
                conn = self._connect()
                cur = conn.cursor()
                cur.execute("DELETE FROM chunks")
                cur.execute("DELETE FROM chunks_fts")
                conn.commit()
                break
            except sqlite3.DatabaseError as ex:
                if attempt == 0 and self._is_malformed_error(ex):
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    self._recover_malformed_db(reason=str(ex))
                    continue
                raise
            finally:
                if conn is not None:
                    conn.close()

        logger.warning("LexicalStore cleared")
