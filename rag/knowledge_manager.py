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
from .chroma_util import ChromaUtil, DocumentResults


class KnowledgeManager:
    """
    Main interface for RAG knowledge management
    Handles storage, retrieval, and organization of knowledge chunks
    
    이제 ChromaDB 접근은 ChromaUtil을 통해서만 이루어집니다.
    """

    def __init__(self, chroma_util: ChromaUtil, embedding_service, metadata_db_path: str = None):
        """
        Initialize knowledge manager with required services

        Args:
            chroma_util: ChromaUtil instance for ChromaDB access
            embedding_service: Embedding generation service
            metadata_db_path: Path to SQLite metadata database
        """
        self.chroma_util = chroma_util
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
        # Store agent metadata in SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO agents (agent_id, agent_name, agent_type) VALUES (?, ?, ?)",
            (agent_id, agent_name, agent_type)
        )
        conn.commit()
        conn.close()

        # Create vector collection using ChromaUtil with metadata
        from datetime import datetime
        collection_metadata = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "created_at": datetime.now().isoformat()
        }
        self.chroma_util.create_collection(agent_id, metadata=collection_metadata)

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

        # Prepare metadata for ChromaDB
        chroma_metadata = {
            "knowledge_id": knowledge_id,
            "metadata": json.dumps(metadata or {}),
            "tags": json.dumps(tags or []),
            "source": source or "",
            "created_at": datetime.now().isoformat()
        }

        # Store in ChromaDB using ChromaUtil
        try:
            collection = self.chroma_util.client.get_collection(name=agent_id)
            collection.add(
                ids=[knowledge_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata]
            )
        except Exception as e:
            raise Exception(f"Failed to store knowledge in ChromaDB: {e}")

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

        # Search using ChromaUtil
        try:
            collection = self.chroma_util.client.get_collection(name=agent_id)
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
        except Exception as e:
            raise Exception(f"Failed to search knowledge in ChromaDB: {e}")

        # Format results
        results = []
        if search_results['ids'] and len(search_results['ids']) > 0:
            for i in range(len(search_results['ids'][0])):
                distance = search_results['distances'][0][i]
                
                # Convert distance to similarity
                if distance < 0:
                    similarity = abs(distance)
                elif distance <= 2.0:
                    similarity = 1.0 - (distance / 2.0)
                else:
                    similarity = 1.0 / (1.0 + distance)
                
                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    chroma_metadata = search_results['metadatas'][0][i]
                    
                    # Parse JSON strings back to objects
                    try:
                        metadata = json.loads(chroma_metadata.get('metadata', '{}'))
                    except:
                        metadata = {}
                    
                    try:
                        tags = json.loads(chroma_metadata.get('tags', '[]'))
                    except:
                        tags = []
                    
                    results.append({
                        "knowledge_id": search_results['ids'][0][i],
                        "content": search_results['documents'][0][i],
                        "metadata": metadata,
                        "tags": tags,
                        "source": chroma_metadata.get('source'),
                        "similarity_score": similarity
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
        # Delete from ChromaDB using ChromaUtil
        try:
            collection = self.chroma_util.client.get_collection(name=agent_id)
            collection.delete(ids=[knowledge_id])
        except Exception as e:
            print(f"Warning: Failed to delete from ChromaDB: {e}")
            return False

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

    def delete_agent_collection(self, agent_id: str) -> bool:
        """
        Delete entire agent collection
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if deletion was successful
        """
        # Delete from ChromaDB
        success = self.chroma_util.delete_collection(agent_id)
        
        if success:
            # Delete from SQLite
            conn = sqlite3.connect(self.metadata_db_path)
            conn.execute("DELETE FROM knowledge WHERE agent_id = ?", (agent_id,))
            conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            conn.commit()
            conn.close()
        
        return success

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
                try:
                    all_tags.extend(json.loads(row[0]))
                except:
                    pass

        unique_tags = list(set(all_tags))

        conn.close()

        # Get ChromaDB collection info
        chroma_info = self.chroma_util.get_collection_info(agent_id)

        return {
            "agent_id": agent_id,
            "knowledge_count": knowledge_count,
            "chroma_count": chroma_info.get('count', 0),
            "unique_tags": unique_tags,
            "tag_count": len(unique_tags)
        }

    # ========================================
    # ChromaUtil 기반 편의 메서드들
    # ========================================

    def show_all_agents(self) -> List[str]:
        """
        모든 에이전트(콜렉션) 목록 표시
        
        Returns:
            에이전트 ID 리스트
        """
        return self.chroma_util.show_collections()

    def show_agent_documents(
        self,
        agent_id: str,
        start: int = 0,
        size: int = 10
    ) -> DocumentResults:
        """
        에이전트의 문서들을 표시
        
        Args:
            agent_id: 에이전트 ID
            start: 시작 인덱스
            size: 가져올 문서 개수
            
        Returns:
            DocumentResults 객체
        """
        return self.chroma_util.show_documents(agent_id, start, size)

    def search_agent_knowledge(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> DocumentResults:
        """
        에이전트 지식 검색 (ChromaUtil 사용)
        
        Args:
            agent_id: 에이전트 ID
            query: 검색 쿼리
            limit: 최대 결과 개수
            min_similarity: 최소 유사도
            
        Returns:
            DocumentResults 객체 (필터링 및 체이닝 가능)
        """
        results = self.chroma_util.search_similar(agent_id, query, limit)
        
        if min_similarity > 0:
            results = results.get_similarity_gte(min_similarity)
        
        return results

    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """
        에이전트 상세 정보 조회
        
        Args:
            agent_id: 에이전트 ID
            
        Returns:
            에이전트 정보 딕셔너리
        """
        return self.chroma_util.get_collection_info(agent_id)