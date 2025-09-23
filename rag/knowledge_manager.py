"""
Knowledge Manager Module
Core RAG functionality for knowledge storage and retrieval
"""

import uuid
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .utils.chunking import TextChunker, ChunkConfig


class KnowledgeManager:
    """
    Main interface for RAG knowledge management
    Handles storage, retrieval, and organization of knowledge chunks
    """

    def __init__(self, vector_store, embedding_service, metadata_db_path: str = None):
        """
        Initialize knowledge manager with required services

        Args:
            vector_store: Vector storage implementation
            embedding_service: Embedding generation service
            metadata_db_path: Path to SQLite metadata database
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.metadata_db_path = metadata_db_path or ":memory:"

        # Initialize text chunker with default config
        chunk_config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        self.chunker = TextChunker(chunk_config)

        # Initialize metadata database
        self._init_metadata_db()

    def _init_metadata_db(self):
        """Initialize SQLite database for metadata storage"""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                knowledge_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                tags TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_agent_collection(self, agent_id: str, agent_name: str, agent_type: str):
        """
        Create a new agent collection

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            agent_type: Type/category of the agent
        """
        # Store agent metadata
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO agents (agent_id, agent_name, agent_type) VALUES (?, ?, ?)",
            (agent_id, agent_name, agent_type)
        )
        conn.commit()
        conn.close()

        # Create vector collection
        dimension = self.embedding_service.get_embedding_dimension()
        self.vector_store.create_collection(agent_id, dimension)

    def store_knowledge(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> str:
        """
        Store knowledge content for an agent

        Args:
            agent_id: Agent identifier
            content: Text content to store
            metadata: Optional metadata dictionary
            tags: Optional list of tags
            source: Optional source identifier

        Returns:
            knowledge_id: Unique identifier for the stored knowledge
        """
        knowledge_id = str(uuid.uuid4())

        # Generate embedding
        embedding = self.embedding_service.generate_embedding(content)

        # Store in vector database
        self.vector_store.add_vectors(
            collection_name=agent_id,
            ids=[knowledge_id],
            embeddings=[embedding],
            metadatas=[{
                "metadata": metadata or {},
                "tags": tags or [],
                "source": source
            }],
            documents=[content]
        )

        # Store metadata in SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute(
            "INSERT INTO knowledge (knowledge_id, agent_id, content, metadata, tags, source) VALUES (?, ?, ?, ?, ?, ?)",
            (
                knowledge_id,
                agent_id,
                content,
                json.dumps(metadata or {}),
                json.dumps(tags or []),
                source
            )
        )
        conn.commit()
        conn.close()

        return knowledge_id

    def load_knowledge(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge based on semantic similarity

        Args:
            agent_id: Agent identifier
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of knowledge entries with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)

        # Search vector store
        search_results = self.vector_store.search_vectors(
            collection_name=agent_id,
            query_embedding=query_embedding,
            limit=limit
        )

        # Format results
        results = []
        for result in search_results:
            # Filter by similarity threshold
            if result["similarity_score"] >= similarity_threshold:
                results.append({
                    "knowledge_id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"].get("metadata", {}),
                    "tags": result["metadata"].get("tags", []),
                    "source": result["metadata"].get("source"),
                    "similarity_score": result["similarity_score"]
                })

        return results

    def delete_knowledge(self, agent_id: str, knowledge_id: str) -> bool:
        """
        Delete specific knowledge entry

        Args:
            agent_id: Agent identifier
            knowledge_id: Knowledge entry identifier

        Returns:
            True if deletion was successful
        """
        # Delete from vector store
        self.vector_store.delete_vectors(agent_id, [knowledge_id])

        # Delete from metadata database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute(
            "DELETE FROM knowledge WHERE knowledge_id = ? AND agent_id = ?",
            (knowledge_id, agent_id)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get statistics for an agent's knowledge collection

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with collection statistics
        """
        conn = sqlite3.connect(self.metadata_db_path)

        # Get knowledge count
        cursor = conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE agent_id = ?",
            (agent_id,)
        )
        knowledge_count = cursor.fetchone()[0]

        # Get unique tags
        cursor = conn.execute(
            "SELECT tags FROM knowledge WHERE agent_id = ?",
            (agent_id,)
        )
        all_tags = []
        for row in cursor.fetchall():
            if row[0]:
                all_tags.extend(json.loads(row[0]))

        unique_tags = list(set(all_tags))

        conn.close()

        return {
            "agent_id": agent_id,
            "knowledge_count": knowledge_count,
            "unique_tags": unique_tags,
            "tag_count": len(unique_tags)
        }