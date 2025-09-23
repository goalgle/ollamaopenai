# RAG Architecture Design: Agent-Specific Knowledge System

## System Overview

A Retrieval-Augmented Generation (RAG) system integrated with the OpenAI SDK + Ollama agents, providing each agent with persistent, searchable knowledge storage through vector databases. Each agent maintains isolated knowledge bases identifiable by Agent ID.

## Architecture Components

### 1. RAG System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  Math Agent    │  Coding Agent   │  Creative Agent │ Custom...  │
│  ID: math-001  │  ID: code-001   │  ID: write-001  │ ID: xxx... │
├─────────────────────────────────────────────────────────────────┤
│                     RAG Middleware                              │
├─────────────────────────────────────────────────────────────────┤
│ Knowledge Manager │ Retrieval Engine │ Embedding Service        │
├─────────────────────────────────────────────────────────────────┤
│                   Vector Database Layer                         │
├─────────────────────────────────────────────────────────────────┤
│ Agent Collections │ Metadata Store  │ Embedding Store          │
├─────────────────────────────────────────────────────────────────┤
│                   Storage Backends                              │
├─────────────────────────────────────────────────────────────────┤
│    ChromaDB     │    Qdrant       │    FAISS       │ SQLite    │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Agent-Specific Knowledge Isolation

Each agent maintains separate knowledge spaces:

```
Vector Database Structure:
├── Collections/
│   ├── agent-math-001/
│   │   ├── mathematical_concepts.db
│   │   ├── problem_solutions.db
│   │   └── learning_patterns.db
│   ├── agent-code-001/
│   │   ├── code_patterns.db
│   │   ├── debugging_solutions.db
│   │   └── api_documentation.db
│   ├── agent-write-001/
│   │   ├── writing_styles.db
│   │   ├── character_profiles.db
│   │   └── plot_structures.db
│   └── agent-custom-xxx/
│       └── domain_knowledge.db
```

## Core Components Design

### 1. Knowledge Manager

```python
@dataclass
class KnowledgeEntry:
    id: str
    agent_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    source: str
    relevance_score: Optional[float] = None

class KnowledgeManager:
    """Manages agent-specific knowledge storage and retrieval"""

    def store_knowledge(
        self,
        agent_id: str,
        content: str,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        source: str = "manual"
    ) -> str:
        """Store knowledge for specific agent"""

    def load_knowledge(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[KnowledgeEntry]:
        """Retrieve relevant knowledge for agent"""

    def update_knowledge(
        self,
        knowledge_id: str,
        content: str = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> bool:
        """Update existing knowledge entry"""

    def delete_knowledge(
        self,
        agent_id: str,
        knowledge_id: str = None,
        tags: List[str] = None
    ) -> bool:
        """Delete knowledge entries"""
```

### 2. Embedding Service

```python
class EmbeddingService:
    """Handles text embedding generation using local models"""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.ollama_client = self._setup_ollama_client()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama embedding model"""

    def batch_generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
```

### 3. Vector Database Abstraction

```python
class VectorStore(ABC):
    """Abstract interface for vector database implementations"""

    @abstractmethod
    def create_collection(self, agent_id: str) -> bool:
        """Create isolated collection for agent"""

    @abstractmethod
    def store_vector(
        self,
        agent_id: str,
        entry: KnowledgeEntry
    ) -> bool:
        """Store knowledge entry with vector"""

    @abstractmethod
    def search_vectors(
        self,
        agent_id: str,
        query_embedding: List[float],
        limit: int,
        threshold: float
    ) -> List[KnowledgeEntry]:
        """Search for similar vectors in agent collection"""

    @abstractmethod
    def delete_vectors(
        self,
        agent_id: str,
        entry_ids: List[str]
    ) -> bool:
        """Delete specific vectors from agent collection"""

class ChromaDBStore(VectorStore):
    """ChromaDB implementation for vector storage"""

class QdrantStore(VectorStore):
    """Qdrant implementation for vector storage"""

class FAISSStore(VectorStore):
    """FAISS implementation for vector storage"""
```

## RAG Integration with Agents

### 1. Enhanced Agent Base Class

```python
class RAGAgent(Agent):
    """Enhanced agent with RAG capabilities"""

    def __init__(
        self,
        agent_id: str,
        name: str,
        instructions: str,
        model: str,
        knowledge_manager: KnowledgeManager,
        **kwargs
    ):
        super().__init__(name, instructions, model, **kwargs)
        self.agent_id = agent_id
        self.knowledge_manager = knowledge_manager
        self.rag_enabled = True

    def process_with_rag(
        self,
        query: str,
        use_knowledge: bool = True,
        store_interaction: bool = True
    ) -> str:
        """Process query with RAG enhancement"""

        context = ""
        if use_knowledge:
            # Retrieve relevant knowledge
            knowledge = self.knowledge_manager.load_knowledge(
                agent_id=self.agent_id,
                query=query,
                limit=5
            )
            context = self._format_knowledge_context(knowledge)

        # Enhance prompt with context
        enhanced_prompt = self._build_rag_prompt(query, context)

        # Process with original agent
        response = super().process(enhanced_prompt)

        if store_interaction:
            # Store this interaction for future reference
            self.knowledge_manager.store_knowledge(
                agent_id=self.agent_id,
                content=f"Query: {query}\nResponse: {response}",
                metadata={
                    "interaction_type": "conversation",
                    "timestamp": datetime.now().isoformat()
                },
                tags=["conversation", "learned_response"],
                source="agent_interaction"
            )

        return response

    def learn_from_document(
        self,
        document: str,
        metadata: Dict[str, Any] = None,
        chunk_size: int = 1000
    ) -> int:
        """Learn from document by chunking and storing"""

        chunks = self._chunk_document(document, chunk_size)
        stored_count = 0

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_chunk": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }

            entry_id = self.knowledge_manager.store_knowledge(
                agent_id=self.agent_id,
                content=chunk,
                metadata=chunk_metadata,
                tags=["document", "learned_content"],
                source="document_learning"
            )

            if entry_id:
                stored_count += 1

        return stored_count
```

### 2. Agent Factory Enhancement

```python
class RAGAgentFactory(AgentFactory):
    """Enhanced factory for creating RAG-enabled agents"""

    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager

    def create_rag_math_tutor(
        self,
        agent_id: str = None,
        model: str = "llama3.2",
        preload_knowledge: bool = True
    ) -> RAGAgent:
        """Create RAG-enabled math tutor"""

        if agent_id is None:
            agent_id = f"math-{uuid.uuid4().hex[:8]}"

        base_agent = super().create_math_tutor(model)

        rag_agent = RAGAgent(
            agent_id=agent_id,
            name=base_agent.name,
            instructions=base_agent.instructions,
            model=base_agent.model,
            knowledge_manager=self.knowledge_manager,
            model_settings=base_agent.model_settings
        )

        if preload_knowledge:
            self._preload_math_knowledge(rag_agent)

        return rag_agent

    def create_rag_coding_assistant(
        self,
        agent_id: str = None,
        model: str = "llama3.2",
        preload_knowledge: bool = True
    ) -> RAGAgent:
        """Create RAG-enabled coding assistant"""

        if agent_id is None:
            agent_id = f"code-{uuid.uuid4().hex[:8]}"

        base_agent = super().create_coding_assistant(model)

        rag_agent = RAGAgent(
            agent_id=agent_id,
            name=base_agent.name,
            instructions=base_agent.instructions,
            model=base_agent.model,
            knowledge_manager=self.knowledge_manager,
            model_settings=base_agent.model_settings
        )

        if preload_knowledge:
            self._preload_coding_knowledge(rag_agent)

        return rag_agent
```

## Database Schema Design

### 1. Knowledge Entries Table

```sql
CREATE TABLE knowledge_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSON,
    tags JSON,
    source TEXT DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_model TEXT,
    relevance_score REAL,

    INDEX idx_agent_id (agent_id),
    INDEX idx_tags (tags),
    INDEX idx_source (source),
    INDEX idx_created_at (created_at),
    UNIQUE (agent_id, content_hash)
);
```

### 2. Agent Collections Table

```sql
CREATE TABLE agent_collections (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    collection_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    entry_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,

    INDEX idx_agent_type (agent_type),
    INDEX idx_last_accessed (last_accessed)
);
```

### 3. Embedding Vectors Table

```sql
CREATE TABLE embedding_vectors (
    id TEXT PRIMARY KEY,
    knowledge_entry_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (knowledge_entry_id) REFERENCES knowledge_entries(id),
    INDEX idx_agent_id (agent_id),
    INDEX idx_embedding_model (embedding_model)
);
```

## API Specifications

### 1. Knowledge Management API

```python
# Store knowledge for specific agent
POST /api/agents/{agent_id}/knowledge
{
    "content": "Mathematical concept explanation...",
    "metadata": {
        "topic": "calculus",
        "difficulty": "intermediate"
    },
    "tags": ["calculus", "derivatives", "concepts"],
    "source": "textbook"
}

# Load knowledge for agent
GET /api/agents/{agent_id}/knowledge?query=derivatives&limit=5&threshold=0.7
Response: {
    "results": [
        {
            "id": "knowledge_123",
            "content": "Derivatives represent rate of change...",
            "relevance_score": 0.95,
            "metadata": {...},
            "tags": ["calculus", "derivatives"]
        }
    ],
    "total": 15,
    "query_embedding_time": 0.05,
    "search_time": 0.02
}

# Update knowledge entry
PUT /api/agents/{agent_id}/knowledge/{knowledge_id}
{
    "content": "Updated content...",
    "metadata": {...},
    "tags": [...]
}

# Delete knowledge entries
DELETE /api/agents/{agent_id}/knowledge?tags=outdated&confirm=true
```

### 2. RAG Processing API

```python
# Process query with RAG
POST /api/agents/{agent_id}/rag-query
{
    "query": "How do I calculate derivatives?",
    "use_knowledge": true,
    "store_interaction": true,
    "max_context_entries": 5,
    "similarity_threshold": 0.7
}

Response: {
    "response": "Based on previous learning materials...",
    "knowledge_used": [
        {
            "id": "knowledge_123",
            "relevance_score": 0.95,
            "content_preview": "Derivatives represent..."
        }
    ],
    "processing_time": 1.25,
    "stored_interaction_id": "interaction_456"
}

# Learn from document
POST /api/agents/{agent_id}/learn
{
    "document": "Mathematical textbook content...",
    "metadata": {
        "source": "calculus_textbook",
        "chapter": "derivatives"
    },
    "chunk_size": 1000,
    "tags": ["textbook", "calculus"]
}

Response: {
    "chunks_processed": 25,
    "chunks_stored": 25,
    "processing_time": 3.45,
    "knowledge_ids": ["knowledge_124", "knowledge_125", ...]
}
```

## Implementation Architecture

### 1. Technology Stack

**Vector Databases:**
- **Primary**: ChromaDB (lightweight, embedded)
- **Alternative**: Qdrant (production-scale)
- **Local**: FAISS (in-memory, fast)

**Embedding Models:**
- **Local**: nomic-embed-text (via Ollama)
- **Alternative**: sentence-transformers models
- **Fallback**: OpenAI text-embedding-ada-002

**Storage:**
- **Metadata**: SQLite (development) / PostgreSQL (production)
- **Documents**: Local filesystem / S3-compatible storage
- **Cache**: Redis (optional, for performance)

### 2. Directory Structure

```
src/
├── rag/
│   ├── __init__.py
│   ├── knowledge_manager.py
│   ├── embedding_service.py
│   ├── vector_stores/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chromadb_store.py
│   │   ├── qdrant_store.py
│   │   └── faiss_store.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── rag_agent.py
│   │   └── rag_factory.py
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py
│       ├── preprocessing.py
│       └── similarity.py
├── api/
│   ├── __init__.py
│   ├── knowledge_routes.py
│   └── rag_routes.py
└── storage/
    ├── __init__.py
    ├── database.py
    └── migrations/
```

### 3. Configuration Management

```python
@dataclass
class RAGConfig:
    # Vector Database Configuration
    vector_store_type: str = "chromadb"  # chromadb, qdrant, faiss
    vector_store_path: str = "./data/vector_db"
    collection_prefix: str = "agent-"

    # Embedding Configuration
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    batch_size: int = 32

    # Retrieval Configuration
    default_similarity_threshold: float = 0.7
    default_retrieval_limit: int = 5
    max_context_length: int = 4000

    # Storage Configuration
    metadata_db_path: str = "./data/metadata.db"
    document_storage_path: str = "./data/documents"

    # Performance Configuration
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    parallel_processing: bool = True
    max_workers: int = 4
```

## Performance Considerations

### 1. Optimization Strategies

**Embedding Optimization:**
- Batch embedding generation for efficiency
- Caching embeddings to avoid recomputation
- Async processing for large documents
- Model quantization for faster inference

**Retrieval Optimization:**
- Index optimization for vector similarity search
- Hierarchical clustering for large knowledge bases
- Query expansion and refinement
- Result caching with TTL

**Storage Optimization:**
- Compression for vector storage
- Partitioning by agent_id for performance
- Connection pooling for database access
- Lazy loading of vector data

### 2. Scalability Design

**Horizontal Scaling:**
- Agent collections can be distributed across nodes
- Vector databases support sharding
- Independent scaling of embedding service
- Load balancing for RAG queries

**Vertical Scaling:**
- Memory-mapped vector indices
- GPU acceleration for embeddings
- SSD storage for vector databases
- Connection pooling and caching

## Security and Privacy

### 1. Data Isolation

**Agent Separation:**
- Each agent has isolated knowledge collections
- Cross-agent knowledge access controls
- Agent-specific access tokens
- Audit logging for knowledge access

**Encryption:**
- At-rest encryption for sensitive knowledge
- In-transit encryption for API communications
- Key management for agent-specific encryption
- Secure deletion of knowledge entries

### 2. Access Control

```python
class KnowledgeAccessControl:
    """Access control for agent knowledge"""

    def can_access_knowledge(
        self,
        requesting_agent_id: str,
        target_agent_id: str,
        knowledge_id: str
    ) -> bool:
        """Check if agent can access specific knowledge"""

    def can_share_knowledge(
        self,
        source_agent_id: str,
        target_agent_id: str,
        knowledge_tags: List[str]
    ) -> bool:
        """Check if knowledge can be shared between agents"""
```

This RAG architecture design provides a comprehensive foundation for implementing agent-specific knowledge management with vector databases, ensuring scalability, performance, and security while maintaining clear separation between different agents' knowledge bases.