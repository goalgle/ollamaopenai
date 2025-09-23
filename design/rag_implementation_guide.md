# RAG Implementation Guide: Vector Database & Agent Knowledge System

## Implementation Overview

This guide provides step-by-step implementation for the RAG system with agent-specific knowledge management using vector databases and Ollama embeddings.

## Project Structure

```
src/
├── rag/
│   ├── __init__.py
│   ├── knowledge_manager.py       # Core knowledge management
│   ├── embedding_service.py       # Ollama-based embeddings
│   ├── vector_stores/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract vector store interface
│   │   ├── chromadb_store.py     # ChromaDB implementation
│   │   ├── faiss_store.py        # FAISS implementation
│   │   └── qdrant_store.py       # Qdrant implementation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── rag_agent.py          # RAG-enhanced agent class
│   │   └── rag_factory.py        # Factory for RAG agents
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── chunking.py           # Text chunking strategies
│   │   ├── preprocessing.py      # Document preprocessing
│   │   └── similarity.py         # Similarity calculations
│   └── storage/
│       ├── __init__.py
│       ├── metadata_db.py        # SQLite metadata storage
│       └── migrations/
│           └── 001_initial.sql
├── api/
│   ├── __init__.py
│   ├── knowledge_routes.py       # Knowledge management API
│   └── rag_routes.py            # RAG processing API
├── examples/
│   ├── rag_basic_usage.py
│   ├── rag_document_learning.py
│   └── rag_interactive_chat.py
└── tests/
    ├── test_rag_system.py
    └── test_vector_stores.py
```

## Phase 1: Core Infrastructure Implementation

### 1.1 Embedding Service with Ollama (`src/rag/embedding_service.py`)

```python
import requests
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class EmbeddingConfig:
    model_name: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    batch_size: int = 32
    timeout_seconds: int = 30
    max_retries: int = 3
    normalize_embeddings: bool = True

class EmbeddingService:
    """Handles text embedding generation using Ollama models"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.logger = logging.getLogger(__name__)
        self._validate_ollama_connection()

    def _validate_ollama_connection(self) -> bool:
        """Validate Ollama server connection and model availability"""
        try:
            response = requests.get(
                f"{self.config.ollama_base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if not any(self.config.model_name in name for name in model_names):
                self.logger.warning(
                    f"Embedding model '{self.config.model_name}' not found. "
                    f"Available models: {model_names}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Ollama server not available: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/embeddings",
                json={
                    "model": self.config.model_name,
                    "prompt": text
                },
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()

            embedding = response.json().get('embedding', [])

            if not embedding:
                raise ValueError("Empty embedding returned from Ollama")

            if self.config.normalize_embeddings:
                embedding = self._normalize_vector(embedding)

            return embedding

        except requests.RequestException as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def batch_generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching"""
        if not texts:
            return []

        embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if show_progress:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            batch_embeddings = []
            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"Failed to embed text: {text[:50]}... Error: {e}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 768)  # Default dimension

            embeddings.extend(batch_embeddings)

        return embeddings

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Async version of embedding generation"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.config.model_name,
                        "prompt": text
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    embedding = data.get('embedding', [])
                    if not embedding:
                        raise ValueError("Empty embedding returned")

                    if self.config.normalize_embeddings:
                        embedding = self._normalize_vector(embedding)

                    return embedding

            except Exception as e:
                self.logger.error(f"Async embedding generation failed: {e}")
                raise RuntimeError(f"Failed to generate embedding: {e}")

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + distance))

        elif metric == "dot_product":
            # Dot product similarity
            return float(np.dot(vec1, vec2))

        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        vec = np.array(vector)
        norm = np.linalg.norm(vec)

        if norm == 0:
            return vector

        return (vec / norm).tolist()

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings from the model"""
        try:
            sample_embedding = self.generate_embedding("test")
            return len(sample_embedding)
        except Exception as e:
            self.logger.error(f"Failed to get embedding dimension: {e}")
            return 768  # Default for nomic-embed-text
```

### 1.2 Vector Store Abstraction (`src/rag/vector_stores/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VectorSearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    embedding: Optional[List[float]] = None

class VectorStore(ABC):
    """Abstract base class for vector database implementations"""

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new collection for storing vectors"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection"""
        pass

    @abstractmethod
    def add_vectors(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> bool:
        """Add vectors to collection"""
        pass

    @abstractmethod
    def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def get_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[VectorSearchResult]:
        """Get vectors by IDs"""
        pass

    @abstractmethod
    def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> bool:
        """Update existing vectors"""
        pass

    @abstractmethod
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete vectors by IDs"""
        pass

    @abstractmethod
    def count_vectors(self, collection_name: str) -> int:
        """Count vectors in collection"""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections"""
        pass
```

### 1.3 ChromaDB Implementation (`src/rag/vector_stores/chromadb_store.py`)

```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from .base import VectorStore, VectorSearchResult

class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector storage"""

    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        self.logger = logging.getLogger(__name__)

        if host and port:
            # Remote ChromaDB instance
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local persistent ChromaDB
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create ChromaDB collection"""
        try:
            self.client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            self.logger.info(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.info(f"Collection {collection_name} already exists")
                return True

            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete ChromaDB collection"""
        try:
            self.client.delete_collection(name=collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    def add_vectors(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> bool:
        """Add vectors to ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)

            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            self.logger.debug(f"Added {len(ids)} vectors to {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add vectors to {collection_name}: {e}")
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search ChromaDB collection for similar vectors"""
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where
            )

            search_results = []

            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    search_results.append(VectorSearchResult(
                        id=doc_id,
                        content=results['documents'][0][i] if results['documents'] else "",
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        similarity_score=1.0 - results['distances'][0][i] if results['distances'] else 1.0,
                        embedding=results['embeddings'][0][i] if results['embeddings'] else None
                    ))

            return search_results

        except Exception as e:
            self.logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return []

    def get_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[VectorSearchResult]:
        """Get specific vectors by IDs"""
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.get(
                ids=ids,
                include=['documents', 'metadatas', 'embeddings']
            )

            search_results = []

            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    search_results.append(VectorSearchResult(
                        id=doc_id,
                        content=results['documents'][i] if results['documents'] else "",
                        metadata=results['metadatas'][i] if results['metadatas'] else {},
                        similarity_score=1.0,  # Perfect match for exact ID lookup
                        embedding=results['embeddings'][i] if results['embeddings'] else None
                    ))

            return search_results

        except Exception as e:
            self.logger.error(f"Failed to get vectors from {collection_name}: {e}")
            return []

    def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> bool:
        """Update vectors in ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)

            update_params = {"ids": ids}
            if embeddings:
                update_params["embeddings"] = embeddings
            if metadatas:
                update_params["metadatas"] = metadatas
            if documents:
                update_params["documents"] = documents

            collection.update(**update_params)

            self.logger.debug(f"Updated {len(ids)} vectors in {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update vectors in {collection_name}: {e}")
            return False

    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete vectors from ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=ids)

            self.logger.debug(f"Deleted {len(ids)} vectors from {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            return False

    def count_vectors(self, collection_name: str) -> int:
        """Count vectors in ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()

        except Exception as e:
            self.logger.error(f"Failed to count vectors in {collection_name}: {e}")
            return 0

    def list_collections(self) -> List[str]:
        """List all ChromaDB collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]

        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []
```

### 1.4 Knowledge Manager Core (`src/rag/knowledge_manager.py`)

```python
import uuid
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
import logging
from .embedding_service import EmbeddingService
from .vector_stores.base import VectorStore
from .utils.chunking import TextChunker

@dataclass
class KnowledgeEntry:
    id: str
    agent_id: str
    content: str
    content_hash: str
    metadata: Dict[str, Any]
    tags: List[str]
    source: str
    created_at: datetime
    updated_at: datetime
    embedding_model: str
    relevance_score: Optional[float] = None

class KnowledgeManager:
    """Core knowledge management with vector storage and metadata DB"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        metadata_db_path: str = "./data/metadata.db"
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.metadata_db_path = metadata_db_path
        self.chunker = TextChunker()
        self.logger = logging.getLogger(__name__)

        self._initialize_metadata_db()

    def _initialize_metadata_db(self):
        """Initialize SQLite metadata database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        # Knowledge entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT NOT NULL,
                tags TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embedding_model TEXT NOT NULL,

                UNIQUE(agent_id, content_hash)
            )
        ''')

        # Agent collections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_collections (
                agent_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                entry_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        ''')

        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON knowledge_entries(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON knowledge_entries(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_entries(created_at)')

        conn.commit()
        conn.close()

    def create_agent_collection(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str
    ) -> bool:
        """Create collection for new agent"""
        try:
            # Create vector collection
            embedding_dim = self.embedding_service.get_embedding_dimension()
            collection_name = f"agent-{agent_id}"

            if not self.vector_store.create_collection(collection_name, embedding_dim):
                return False

            # Record in metadata DB
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO agent_collections
                (agent_id, agent_name, agent_type, created_at)
                VALUES (?, ?, ?, ?)
            ''', (agent_id, agent_name, agent_type, datetime.now().isoformat()))

            conn.commit()
            conn.close()

            self.logger.info(f"Created collection for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create agent collection {agent_id}: {e}")
            return False

    def store_knowledge(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        source: str = "manual"
    ) -> str:
        """Store knowledge entry for agent"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Check for existing content
            if self._knowledge_exists(agent_id, content_hash):
                self.logger.info(f"Knowledge already exists for agent {agent_id}")
                return self._get_existing_knowledge_id(agent_id, content_hash)

            # Generate unique ID and embedding
            knowledge_id = str(uuid.uuid4())
            embedding = self.embedding_service.generate_embedding(content)

            # Create knowledge entry
            entry = KnowledgeEntry(
                id=knowledge_id,
                agent_id=agent_id,
                content=content,
                content_hash=content_hash,
                metadata=metadata or {},
                tags=tags or [],
                source=source,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                embedding_model=self.embedding_service.config.model_name
            )

            # Store in vector database
            collection_name = f"agent-{agent_id}"

            success = self.vector_store.add_vectors(
                collection_name=collection_name,
                ids=[knowledge_id],
                embeddings=[embedding],
                metadatas=[{
                    "agent_id": agent_id,
                    "source": source,
                    "tags": json.dumps(tags or []),
                    "created_at": entry.created_at.isoformat(),
                    **metadata or {}
                }],
                documents=[content]
            )

            if not success:
                raise RuntimeError("Failed to store in vector database")

            # Store metadata in SQLite
            self._store_metadata(entry)

            # Update agent collection stats
            self._update_agent_stats(agent_id)

            self.logger.info(f"Stored knowledge {knowledge_id} for agent {agent_id}")
            return knowledge_id

        except Exception as e:
            self.logger.error(f"Failed to store knowledge for agent {agent_id}: {e}")
            raise RuntimeError(f"Knowledge storage failed: {e}")

    def load_knowledge(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        tags_filter: Optional[List[str]] = None
    ) -> List[KnowledgeEntry]:
        """Load relevant knowledge for agent based on query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)

            # Search vector database
            collection_name = f"agent-{agent_id}"

            where_filter = {}
            if tags_filter:
                # Note: ChromaDB filtering syntax may vary
                where_filter["tags"] = {"$in": tags_filter}

            search_results = self.vector_store.search_vectors(
                collection_name=collection_name,
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more to filter by threshold
                where=where_filter
            )

            # Filter by similarity threshold and convert to KnowledgeEntry
            relevant_entries = []

            for result in search_results:
                if result.similarity_score >= similarity_threshold:
                    # Load full metadata from SQLite
                    entry = self._load_metadata(result.id)
                    if entry:
                        entry.relevance_score = result.similarity_score
                        relevant_entries.append(entry)

                if len(relevant_entries) >= limit:
                    break

            # Update last accessed time
            self._update_last_accessed(agent_id)

            self.logger.debug(
                f"Loaded {len(relevant_entries)} knowledge entries for agent {agent_id}"
            )
            return relevant_entries

        except Exception as e:
            self.logger.error(f"Failed to load knowledge for agent {agent_id}: {e}")
            return []

    def load_knowledge_by_id(
        self,
        agent_id: str,
        knowledge_ids: List[str]
    ) -> List[KnowledgeEntry]:
        """Load specific knowledge entries by IDs"""
        try:
            collection_name = f"agent-{agent_id}"

            search_results = self.vector_store.get_vectors(
                collection_name=collection_name,
                ids=knowledge_ids
            )

            entries = []
            for result in search_results:
                entry = self._load_metadata(result.id)
                if entry:
                    entries.append(entry)

            return entries

        except Exception as e:
            self.logger.error(f"Failed to load knowledge by IDs for agent {agent_id}: {e}")
            return []

    def _knowledge_exists(self, agent_id: str, content_hash: str) -> bool:
        """Check if knowledge with content hash already exists"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id FROM knowledge_entries WHERE agent_id = ? AND content_hash = ?',
            (agent_id, content_hash)
        )

        result = cursor.fetchone()
        conn.close()

        return result is not None

    def _get_existing_knowledge_id(self, agent_id: str, content_hash: str) -> str:
        """Get ID of existing knowledge entry"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id FROM knowledge_entries WHERE agent_id = ? AND content_hash = ?',
            (agent_id, content_hash)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else ""

    def _store_metadata(self, entry: KnowledgeEntry):
        """Store knowledge entry metadata in SQLite"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO knowledge_entries
            (id, agent_id, content, content_hash, metadata, tags, source,
             created_at, updated_at, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.id,
            entry.agent_id,
            entry.content,
            entry.content_hash,
            json.dumps(entry.metadata),
            json.dumps(entry.tags),
            entry.source,
            entry.created_at.isoformat(),
            entry.updated_at.isoformat(),
            entry.embedding_model
        ))

        conn.commit()
        conn.close()

    def _load_metadata(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        """Load knowledge entry metadata from SQLite"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM knowledge_entries WHERE id = ?',
            (knowledge_id,)
        )

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        return KnowledgeEntry(
            id=result[0],
            agent_id=result[1],
            content=result[2],
            content_hash=result[3],
            metadata=json.loads(result[4]),
            tags=json.loads(result[5]),
            source=result[6],
            created_at=datetime.fromisoformat(result[7]),
            updated_at=datetime.fromisoformat(result[8]),
            embedding_model=result[9]
        )

    def _update_agent_stats(self, agent_id: str):
        """Update agent collection statistics"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        # Count entries for agent
        cursor.execute(
            'SELECT COUNT(*) FROM knowledge_entries WHERE agent_id = ?',
            (agent_id,)
        )
        count = cursor.fetchone()[0]

        # Update agent stats
        cursor.execute('''
            UPDATE agent_collections
            SET entry_count = ?, last_accessed = ?
            WHERE agent_id = ?
        ''', (count, datetime.now().isoformat(), agent_id))

        conn.commit()
        conn.close()

    def _update_last_accessed(self, agent_id: str):
        """Update last accessed time for agent"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE agent_collections
            SET last_accessed = ?
            WHERE agent_id = ?
        ''', (datetime.now().isoformat(), agent_id))

        conn.commit()
        conn.close()
```

### 1.5 Text Chunking Utilities (`src/rag/utils/chunking.py`)

```python
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: str = "recursive"  # recursive, fixed, semantic
    separators: List[str] = None
    keep_separator: bool = True

class TextChunker:
    """Text chunking utilities for document processing"""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        if self.config.separators is None:
            self.config.separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                " ",     # Words
                ""       # Characters
            ]

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[str]:
        """Chunk text using specified strategy"""

        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap
        strategy = strategy or self.config.chunk_strategy

        if strategy == "fixed":
            return self._fixed_chunk(text, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            return self._semantic_chunk(text, chunk_size, chunk_overlap)
        else:  # recursive (default)
            return self._recursive_chunk(text, chunk_size, chunk_overlap)

    def _recursive_chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Recursive chunking preserving structure"""

        def _split_text(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]

            separator = separators[0]
            remaining_separators = separators[1:]

            if separator == "":
                # Character-level split
                return list(text)

            splits = text.split(separator)

            # Keep separator if configured
            if self.config.keep_separator and separator != "":
                result = []
                for i, split in enumerate(splits[:-1]):
                    result.append(split + separator)
                if splits[-1]:  # Add last split if not empty
                    result.append(splits[-1])
                splits = result

            # Recursively split large chunks
            final_chunks = []
            for split in splits:
                if len(split) <= chunk_size:
                    final_chunks.append(split)
                else:
                    # Split further using remaining separators
                    sub_chunks = _split_text(split, remaining_separators)
                    final_chunks.extend(sub_chunks)

            return final_chunks

        # Initial split
        chunks = _split_text(text, self.config.separators)

        # Merge small chunks and handle overlap
        return self._merge_chunks(chunks, chunk_size, chunk_overlap)

    def _fixed_chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Fixed-size chunking with overlap"""

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():  # Skip empty chunks
                chunks.append(chunk)

            # Move start position considering overlap
            start = end - chunk_overlap

            if start >= len(text):
                break

        return chunks

    def _semantic_chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Semantic chunking based on sentences and paragraphs"""

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Handle overlap by keeping last few sentences
                if chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _merge_chunks(
        self,
        chunks: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Merge small chunks while respecting size limits"""

        merged_chunks = []
        current_chunk = ""

        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue

            # If chunk is too large, split it
            if len(chunk) > chunk_size:
                # Add current chunk if it exists
                if current_chunk:
                    merged_chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split large chunk
                large_chunk_splits = self._fixed_chunk(chunk, chunk_size, chunk_overlap)
                merged_chunks.extend(large_chunk_splits)
                continue

            # Check if we can add this chunk to current
            potential_length = len(current_chunk) + len(chunk) + 1  # +1 for space

            if potential_length <= chunk_size:
                # Add to current chunk
                current_chunk += " " + chunk if current_chunk else chunk
            else:
                # Start new chunk
                if current_chunk:
                    merged_chunks.append(current_chunk.strip())

                # Handle overlap
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + " " + chunk
                else:
                    current_chunk = chunk

        # Add final chunk
        if current_chunk.strip():
            merged_chunks.append(current_chunk.strip())

        return merged_chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= overlap_size:
            return text

        # Try to break at word boundaries
        overlap_text = text[-overlap_size:]

        # Find first space to avoid cutting words
        first_space = overlap_text.find(' ')
        if first_space > 0:
            overlap_text = overlap_text[first_space + 1:]

        return overlap_text

    def chunk_document_with_metadata(
        self,
        document: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Chunk document and return with metadata for each chunk"""

        chunks = self.chunk_text(document, chunk_size, chunk_overlap)

        chunk_metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            }
            chunk_metadata_list.append(chunk_metadata)

        return chunk_metadata_list
```

This implementation guide provides the core infrastructure for the RAG system. The next phases would include:

**Phase 2: RAG Agent Integration**
- Enhanced agent classes with RAG capabilities
- Factory patterns for creating RAG-enabled agents

**Phase 3: API Layer**
- REST API endpoints for knowledge management
- RAG processing endpoints

**Phase 4: Testing and Examples**
- Comprehensive test suite
- Usage examples and tutorials

The design emphasizes modularity, allowing for easy swapping of vector databases and embedding models while maintaining a consistent interface for agent-specific knowledge management.