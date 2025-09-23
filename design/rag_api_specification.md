# RAG API Specification: Agent Knowledge Management

## Core API Modules

### 1. Knowledge Management API

#### Store Knowledge Entry
```python
def store_knowledge(
    agent_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    source: str = "manual",
    chunk_strategy: str = "auto"
) -> str:
    """
    Store knowledge entry for specific agent with automatic embedding generation

    Args:
        agent_id: Unique identifier for the agent
        content: Text content to store (will be embedded)
        metadata: Additional metadata for the entry
        tags: List of tags for categorization
        source: Source of the knowledge (manual, document, interaction, etc.)
        chunk_strategy: How to chunk large content (auto, fixed, semantic)

    Returns:
        str: Unique knowledge entry ID

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        ContentTooLargeError: If content exceeds size limits
        EmbeddingError: If embedding generation fails

    Example:
        knowledge_id = store_knowledge(
            agent_id="math-001",
            content="The derivative of x² is 2x",
            metadata={"topic": "calculus", "difficulty": "basic"},
            tags=["derivatives", "calculus", "basic"],
            source="textbook"
        )
    """
```

#### Load Knowledge by Query
```python
def load_knowledge(
    agent_id: str,
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7,
    include_metadata: bool = True,
    tags_filter: Optional[List[str]] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None
) -> List[KnowledgeEntry]:
    """
    Retrieve relevant knowledge entries for agent based on semantic similarity

    Args:
        agent_id: Agent identifier to search within
        query: Search query (will be embedded for similarity search)
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0.0-1.0)
        include_metadata: Whether to include full metadata in results
        tags_filter: Only return entries with these tags
        date_range: Filter entries by creation date range

    Returns:
        List[KnowledgeEntry]: Ranked list of relevant knowledge entries

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        QueryEmbeddingError: If query embedding fails

    Example:
        results = load_knowledge(
            agent_id="math-001",
            query="how to calculate derivatives",
            limit=3,
            similarity_threshold=0.8,
            tags_filter=["calculus"]
        )
    """
```

#### Load Knowledge by ID
```python
def load_knowledge_by_id(
    agent_id: str,
    knowledge_ids: Union[str, List[str]],
    include_embeddings: bool = False
) -> Union[KnowledgeEntry, List[KnowledgeEntry]]:
    """
    Load specific knowledge entries by their IDs

    Args:
        agent_id: Agent identifier for access control
        knowledge_ids: Single ID or list of knowledge entry IDs
        include_embeddings: Whether to include embedding vectors

    Returns:
        KnowledgeEntry or List[KnowledgeEntry]: Requested entries

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        KnowledgeNotFoundError: If any knowledge_id doesn't exist
        AccessDeniedError: If agent doesn't own the knowledge entries
    """
```

#### Update Knowledge Entry
```python
def update_knowledge(
    agent_id: str,
    knowledge_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    regenerate_embedding: bool = True
) -> bool:
    """
    Update existing knowledge entry

    Args:
        agent_id: Agent identifier for access control
        knowledge_id: ID of knowledge entry to update
        content: New content (will regenerate embedding if provided)
        metadata: Updated metadata (merged with existing)
        tags: Updated tags (replaces existing tags)
        regenerate_embedding: Force embedding regeneration

    Returns:
        bool: Success status

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        KnowledgeNotFoundError: If knowledge_id doesn't exist
        AccessDeniedError: If agent doesn't own the knowledge entry
    """
```

#### Delete Knowledge Entries
```python
def delete_knowledge(
    agent_id: str,
    knowledge_ids: Optional[List[str]] = None,
    tags_filter: Optional[List[str]] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    confirm: bool = False
) -> Dict[str, Any]:
    """
    Delete knowledge entries with various filter options

    Args:
        agent_id: Agent identifier for access control
        knowledge_ids: Specific IDs to delete
        tags_filter: Delete entries with these tags
        date_range: Delete entries within date range
        confirm: Required confirmation for batch deletes

    Returns:
        dict: {
            'deleted_count': int,
            'deleted_ids': List[str],
            'failed_ids': List[str]
        }

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        ConfirmationRequiredError: If batch delete without confirmation
    """
```

### 2. Agent Collection Management API

#### Create Agent Collection
```python
def create_agent_collection(
    agent_id: str,
    agent_name: str,
    agent_type: str,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create isolated knowledge collection for new agent

    Args:
        agent_id: Unique agent identifier
        agent_name: Display name for the agent
        agent_type: Type category (math, coding, creative, custom)
        config: Collection-specific configuration

    Returns:
        bool: Success status

    Raises:
        AgentExistsError: If agent_id already exists
        VectorStoreError: If collection creation fails
    """
```

#### List Agent Collections
```python
def list_agent_collections(
    agent_type: Optional[str] = None,
    include_stats: bool = True
) -> List[Dict[str, Any]]:
    """
    List all agent collections with optional filtering

    Args:
        agent_type: Filter by agent type
        include_stats: Include knowledge entry counts and stats

    Returns:
        List[dict]: Agent collection information

    Example Response:
        [
            {
                'agent_id': 'math-001',
                'agent_name': 'Math Tutor',
                'agent_type': 'math',
                'created_at': '2024-01-15T10:30:00Z',
                'entry_count': 150,
                'last_accessed': '2024-01-20T15:45:00Z'
            }
        ]
    """
```

#### Delete Agent Collection
```python
def delete_agent_collection(
    agent_id: str,
    confirm: bool = False,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Delete entire agent collection and all knowledge

    Args:
        agent_id: Agent identifier to delete
        confirm: Required confirmation for destructive operation
        backup: Create backup before deletion

    Returns:
        dict: {
            'deleted': bool,
            'entries_removed': int,
            'backup_path': Optional[str]
        }

    Raises:
        AgentNotFoundError: If agent_id doesn't exist
        ConfirmationRequiredError: If confirm=False
    """
```

### 3. RAG Processing API

#### Process Query with RAG
```python
def process_rag_query(
    agent_id: str,
    query: str,
    use_knowledge: bool = True,
    store_interaction: bool = True,
    max_context_entries: int = 5,
    similarity_threshold: float = 0.7,
    context_window_size: int = 4000
) -> RAGResponse:
    """
    Process query using RAG with agent-specific knowledge

    Args:
        agent_id: Agent to process query with
        query: User query/question
        use_knowledge: Whether to retrieve and use stored knowledge
        store_interaction: Store this Q&A for future learning
        max_context_entries: Maximum knowledge entries to include
        similarity_threshold: Minimum similarity for knowledge inclusion
        context_window_size: Maximum context length in tokens

    Returns:
        RAGResponse: Enhanced response with knowledge context

    RAGResponse:
        {
            'response': str,
            'knowledge_used': List[KnowledgeEntry],
            'context_entries_count': int,
            'similarity_scores': List[float],
            'processing_time': float,
            'stored_interaction_id': Optional[str],
            'model_used': str,
            'embedding_model_used': str
        }
    """
```

#### Learn from Document
```python
def learn_from_document(
    agent_id: str,
    document: Union[str, bytes, Path],
    document_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    tags: Optional[List[str]] = None,
    preprocessing_options: Optional[Dict[str, Any]] = None
) -> DocumentLearningResult:
    """
    Process and learn from documents (text, PDF, etc.)

    Args:
        agent_id: Agent to store knowledge for
        document: Document content, bytes, or file path
        document_type: text, pdf, markdown, html, docx
        metadata: Document metadata to store with chunks
        chunk_size: Target size for text chunks
        chunk_overlap: Overlap between consecutive chunks
        tags: Tags to apply to all learned entries
        preprocessing_options: Document-specific preprocessing settings

    Returns:
        DocumentLearningResult: Learning statistics and results

    DocumentLearningResult:
        {
            'document_id': str,
            'chunks_processed': int,
            'chunks_stored': int,
            'failed_chunks': int,
            'processing_time': float,
            'knowledge_ids': List[str],
            'metadata': Dict[str, Any],
            'error_details': Optional[List[str]]
        }
    """
```

#### Learn from Conversation
```python
def learn_from_conversation(
    agent_id: str,
    messages: List[Dict[str, str]],
    metadata: Optional[Dict[str, Any]] = None,
    extract_key_insights: bool = True,
    tags: Optional[List[str]] = None
) -> ConversationLearningResult:
    """
    Learn from conversation history

    Args:
        agent_id: Agent to store knowledge for
        messages: List of {'role': 'user'|'assistant', 'content': str}
        metadata: Conversation metadata
        extract_key_insights: Extract and store key insights separately
        tags: Tags for categorization

    Returns:
        ConversationLearningResult: Learning results

    Example:
        result = learn_from_conversation(
            agent_id="math-001",
            messages=[
                {'role': 'user', 'content': 'How do I solve x² + 5x + 6 = 0?'},
                {'role': 'assistant', 'content': 'You can factor this as (x+2)(x+3)=0...'}
            ],
            tags=['quadratic_equations', 'factoring']
        )
    """
```

### 4. Embedding and Similarity API

#### Generate Embeddings
```python
def generate_embedding(
    text: str,
    model: str = "nomic-embed-text",
    normalize: bool = True
) -> List[float]:
    """
    Generate embedding vector for text

    Args:
        text: Input text to embed
        model: Embedding model to use
        normalize: Normalize embedding to unit vector

    Returns:
        List[float]: Embedding vector

    Raises:
        EmbeddingError: If embedding generation fails
        ModelNotFoundError: If embedding model not available
    """
```

#### Batch Generate Embeddings
```python
def batch_generate_embeddings(
    texts: List[str],
    model: str = "nomic-embed-text",
    batch_size: int = 32,
    parallel: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        batch_size: Processing batch size
        parallel: Enable parallel processing

    Returns:
        List[List[float]]: List of embedding vectors
    """
```

#### Calculate Similarity
```python
def calculate_similarity(
    embedding1: List[float],
    embedding2: List[float],
    metric: str = "cosine"
) -> float:
    """
    Calculate similarity between two embeddings

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric (cosine, euclidean, dot_product)

    Returns:
        float: Similarity score (0.0-1.0 for cosine)
    """
```

### 5. Knowledge Analytics API

#### Get Knowledge Statistics
```python
def get_knowledge_stats(
    agent_id: str,
    include_embedding_stats: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for agent's knowledge base

    Args:
        agent_id: Agent identifier
        include_embedding_stats: Include embedding dimension and model info

    Returns:
        dict: {
            'total_entries': int,
            'entries_by_source': Dict[str, int],
            'entries_by_tag': Dict[str, int],
            'date_range': {
                'earliest': datetime,
                'latest': datetime
            },
            'content_stats': {
                'total_characters': int,
                'avg_entry_length': float,
                'longest_entry': int
            },
            'embedding_stats': {
                'model': str,
                'dimension': int,
                'total_vectors': int
            }
        }
    """
```

#### Search Knowledge
```python
def search_knowledge(
    agent_id: str,
    search_params: Dict[str, Any]
) -> SearchResult:
    """
    Advanced search with multiple criteria

    Args:
        agent_id: Agent identifier
        search_params: {
            'query': Optional[str],  # Semantic search
            'tags': Optional[List[str]],  # Tag filter
            'content_filter': Optional[str],  # Text search
            'metadata_filter': Optional[Dict],  # Metadata filter
            'date_range': Optional[Tuple[datetime, datetime]],
            'similarity_threshold': Optional[float],
            'limit': Optional[int],
            'offset': Optional[int]
        }

    Returns:
        SearchResult: {
            'results': List[KnowledgeEntry],
            'total_count': int,
            'query_time': float,
            'filters_applied': List[str]
        }
    """
```

### 6. Knowledge Import/Export API

#### Export Knowledge
```python
def export_knowledge(
    agent_id: str,
    format: str = "json",
    include_embeddings: bool = False,
    filter_params: Optional[Dict[str, Any]] = None
) -> Union[str, bytes]:
    """
    Export agent knowledge in various formats

    Args:
        agent_id: Agent identifier
        format: Export format (json, csv, jsonl, pickle)
        include_embeddings: Include embedding vectors in export
        filter_params: Filter knowledge entries to export

    Returns:
        Serialized knowledge data

    Supported Formats:
        - json: Complete knowledge with metadata
        - csv: Tabular format for analysis
        - jsonl: Streaming JSON lines format
        - pickle: Binary format with embeddings
    """
```

#### Import Knowledge
```python
def import_knowledge(
    agent_id: str,
    data: Union[str, bytes, Path],
    format: str = "json",
    merge_strategy: str = "skip_duplicates",
    regenerate_embeddings: bool = False
) -> ImportResult:
    """
    Import knowledge from external sources

    Args:
        agent_id: Target agent identifier
        data: Data to import (content, bytes, or file path)
        format: Data format (json, csv, jsonl, pickle)
        merge_strategy: How to handle duplicates (skip_duplicates, overwrite, merge_metadata)
        regenerate_embeddings: Force regeneration of embeddings

    Returns:
        ImportResult: {
            'imported_count': int,
            'skipped_count': int,
            'failed_count': int,
            'processing_time': float,
            'imported_ids': List[str],
            'error_details': List[str]
        }
    """
```

## Data Models

### Core Data Structures

```python
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
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    relevance_score: Optional[float] = None

@dataclass
class RAGResponse:
    response: str
    knowledge_used: List[KnowledgeEntry]
    context_entries_count: int
    similarity_scores: List[float]
    processing_time: float
    stored_interaction_id: Optional[str]
    model_used: str
    embedding_model_used: str

@dataclass
class DocumentLearningResult:
    document_id: str
    chunks_processed: int
    chunks_stored: int
    failed_chunks: int
    processing_time: float
    knowledge_ids: List[str]
    metadata: Dict[str, Any]
    error_details: Optional[List[str]] = None

@dataclass
class SearchResult:
    results: List[KnowledgeEntry]
    total_count: int
    query_time: float
    filters_applied: List[str]
    pagination: Optional[Dict[str, int]] = None
```

## Error Handling

### Exception Hierarchy

```python
class RAGError(Exception):
    """Base exception for RAG operations"""

class AgentNotFoundError(RAGError):
    """Agent ID not found"""

class KnowledgeNotFoundError(RAGError):
    """Knowledge entry not found"""

class EmbeddingError(RAGError):
    """Embedding generation failed"""

class VectorStoreError(RAGError):
    """Vector database operation failed"""

class ContentTooLargeError(RAGError):
    """Content exceeds size limits"""

class AccessDeniedError(RAGError):
    """Access denied to resource"""

class ConfirmationRequiredError(RAGError):
    """Destructive operation requires confirmation"""
```

## Configuration and Settings

```python
@dataclass
class RAGAPIConfig:
    # Storage Settings
    vector_store_type: str = "chromadb"
    vector_store_url: str = "sqlite:///./data/vectors.db"
    metadata_db_url: str = "sqlite:///./data/metadata.db"

    # Embedding Settings
    default_embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    batch_embedding_size: int = 32

    # Retrieval Settings
    default_similarity_threshold: float = 0.7
    default_retrieval_limit: int = 5
    max_context_window: int = 4000

    # Performance Settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_operations: int = 10

    # Security Settings
    enable_access_control: bool = True
    require_confirmation_for_deletes: bool = True
    max_content_size_mb: int = 50
```

This comprehensive API specification provides a complete interface for managing agent-specific knowledge with vector databases, enabling sophisticated RAG capabilities while maintaining security and performance.