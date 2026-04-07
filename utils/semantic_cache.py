"""
Semantic Query Caching using SQLite + embeddings similarity.
Caches query embeddings and retrieval results to avoid redundant API calls.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from config.settings import (
    ENABLE_SEMANTIC_CACHE,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SEMANTIC_CACHE_TTL_HOURS,
)


class SemanticCache:
    """LLM-aware cache that hits on semantically similar queries."""

    def __init__(self, db_path: str = SEMANTIC_CACHE_PATH, embedding_dim: int = 1024):
        """
        Initialize semantic cache.
        
        Args:
            db_path: Path to SQLite database
            embedding_dim: Dimension of embeddings (e.g., 1024 for bge-m3)
        """
        if not ENABLE_SEMANTIC_CACHE:
            self.enabled = False
            return

        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.enabled = True

        # Create parent directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_embedding BLOB NOT NULL,
                    results JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON query_cache(created_at)
            """)
            conn.commit()

    def get(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        min_similarity: float = SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Look up cached results for semantically similar queries.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding (1D numpy array)
            min_similarity: Cosine similarity threshold (0.95 default)
            
        Returns:
            Cached results if found and similar, else None
        """
        if not self.enabled:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Fetch all cached embeddings within TTL
                ttl_cutoff = datetime.now() - timedelta(hours=SEMANTIC_CACHE_TTL_HOURS)

                cursor.execute(
                    """
                    SELECT id, query_embedding, results, created_at
                    FROM query_cache
                    WHERE created_at > ?
                    ORDER BY created_at DESC
                    """,
                    (ttl_cutoff,),
                )

                rows = cursor.fetchall()

                best_match = None
                best_similarity = 0

                for row_id, embedding_blob, results_json, created_at in rows:
                    cached_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(
                        query_embedding, cached_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (row_id, results_json, similarity)

                # Check if best match meets threshold
                if best_match and best_match[2] >= min_similarity:
                    row_id, results_json, similarity = best_match
                    
                    # Update last_accessed
                    cursor.execute(
                        """
                        UPDATE query_cache 
                        SET last_accessed = CURRENT_TIMESTAMP 
                        WHERE id = ?
                        """,
                        (row_id,),
                    )
                    conn.commit()

                    results = json.loads(results_json)
                    return results

                return None

        except Exception as e:
            print(f"[WARN] Semantic cache lookup failed: {e}")
            return None

    def set(
        self, query_text: str, query_embedding: np.ndarray, results: List[Dict[str, Any]]
    ) -> bool:
        """
        Cache query results.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding (1D numpy array)
            results: Retrieval results to cache
            
        Returns:
            True if cached successfully, else False
        """
        if not self.enabled:
            return False

        try:
            embedding_blob = query_embedding.astype(np.float32).tobytes()
            results_json = json.dumps(results)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO query_cache (query_text, query_embedding, results)
                    VALUES (?, ?, ?)
                    """,
                    (query_text, embedding_blob, results_json),
                )
                conn.commit()

            return True

        except Exception as e:
            print(f"[WARN] Semantic cache write failed: {e}")
            return False

    def cleanup_expired(self):
        """Remove cache entries older than TTL."""
        if not self.enabled:
            return

        try:
            ttl_cutoff = datetime.now() - timedelta(hours=SEMANTIC_CACHE_TTL_HOURS)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM query_cache WHERE created_at < ?", (ttl_cutoff,)
                )
                deleted = cursor.rowcount
                conn.commit()

            if deleted > 0:
                print(f"[DEBUG] Cleaned up {deleted} expired cache entries")

        except Exception as e:
            print(f"[WARN] Semantic cache cleanup failed: {e}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) == 0 or len(b) == 0:
            return 0.0

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def clear(self):
        """Clear entire cache."""
        if not self.enabled:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM query_cache")
                conn.commit()
                print("[DEBUG] Semantic cache cleared")
        except Exception as e:
            print(f"[WARN] Failed to clear cache: {e}")
