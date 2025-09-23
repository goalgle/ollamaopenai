# RAG and Vector DB Testing Strategy

## Testing Overview

Comprehensive testing strategy for the RAG (Retrieval-Augmented Generation) system and Vector Database components, ensuring reliability, performance, and correctness across all layers of the architecture.

## Testing Architecture

### Test Pyramid Structure

```
                    ┌─────────────────────┐
                    │   E2E Tests         │
                    │   (User Scenarios)  │
                    └─────────────────────┘
                  ┌───────────────────────────┐
                  │   Integration Tests       │
                  │   (Component Interaction) │
                  └───────────────────────────┘
                ┌─────────────────────────────────┐
                │        Unit Tests               │
                │   (Individual Components)       │
                └─────────────────────────────────┘
              ┌───────────────────────────────────────┐
              │         Performance Tests             │
              │   (Load, Stress, Benchmarking)       │
              └───────────────────────────────────────┘
```

### Test Categories

#### 1. Unit Tests (70% of test coverage)
- **Vector Store Operations**: CRUD operations for each vector DB backend
- **Embedding Service**: Embedding generation and similarity calculations
- **Knowledge Manager**: Core knowledge storage and retrieval logic
- **Text Chunking**: Document processing and chunking strategies
- **Agent Isolation**: Verify complete separation between agent collections

#### 2. Integration Tests (20% of test coverage)
- **RAG Workflow**: End-to-end knowledge storage and retrieval
- **Agent-Knowledge Integration**: RAG-enhanced agent functionality
- **Cross-Component**: Vector DB + Embedding Service + Knowledge Manager
- **API Layer**: REST API endpoints and error handling
- **Database Consistency**: Vector DB and metadata DB synchronization

#### 3. End-to-End Tests (5% of test coverage)
- **User Scenarios**: Complete user workflows from agent creation to knowledge usage
- **Multi-Agent Scenarios**: Multiple agents with isolated knowledge bases
- **Document Learning**: Full document ingestion and query workflows
- **Performance Scenarios**: Real-world usage patterns and load

#### 4. Performance Tests (5% of test coverage)
- **Load Testing**: High volume concurrent operations
- **Stress Testing**: System behavior under extreme conditions
- **Benchmark Testing**: Performance comparison across vector DB backends
- **Memory Profiling**: Resource usage optimization

## Test Environment Setup

### Dependencies and Infrastructure

```yaml
test_dependencies:
  core:
    - pytest: ">=7.0.0"
    - pytest-asyncio: ">=0.23.0"
    - pytest-cov: ">=4.0.0"
    - pytest-mock: ">=3.12.0"
    - pytest-benchmark: ">=4.0.0"

  vector_databases:
    - chromadb: ">=0.4.0"
    - qdrant-client: ">=1.6.0"
    - faiss-cpu: ">=1.7.0"

  performance:
    - memory-profiler: ">=0.61.0"
    - psutil: ">=5.9.0"
    - locust: ">=2.17.0"

  utilities:
    - factory-boy: ">=3.3.0"  # Test data generation
    - hypothesis: ">=6.88.0"  # Property-based testing
    - testcontainers: ">=3.7.0"  # Container-based testing
```

### Test Data Management

```python
# Test data factories for consistent test scenarios
@dataclass
class TestDataFactory:
    @staticmethod
    def create_test_agent(agent_id: str = None) -> Dict[str, Any]:
        return {
            'agent_id': agent_id or f"test-{uuid.uuid4().hex[:8]}",
            'agent_name': 'Test Agent',
            'agent_type': 'test',
            'created_at': datetime.now()
        }

    @staticmethod
    def create_knowledge_entry(agent_id: str, content: str = None) -> Dict[str, Any]:
        return {
            'agent_id': agent_id,
            'content': content or f"Test knowledge content {uuid.uuid4().hex[:8]}",
            'metadata': {'test': True, 'topic': 'testing'},
            'tags': ['test', 'sample'],
            'source': 'test_factory'
        }

    @staticmethod
    def create_test_document(size: str = "medium") -> str:
        sizes = {
            "small": 500,    # 500 characters
            "medium": 5000,  # 5K characters
            "large": 50000,  # 50K characters
            "xlarge": 500000 # 500K characters
        }

        char_count = sizes.get(size, 5000)
        return f"Test document content. " * (char_count // 20)
```

## Detailed Test Specifications

### 1. Vector Store Tests

#### ChromaDB Store Tests
```python
class TestChromaDBStore:
    """Test ChromaDB vector store implementation"""

    @pytest.fixture
    def chromadb_store(self, tmp_path):
        """Create temporary ChromaDB instance for testing"""
        return ChromaDBStore(persist_directory=str(tmp_path / "test_chromadb"))

    def test_create_collection(self, chromadb_store):
        """Test collection creation"""
        result = chromadb_store.create_collection("test_collection", dimension=768)
        assert result is True

        # Verify collection exists
        collections = chromadb_store.list_collections()
        assert "test_collection" in collections

    def test_add_vectors(self, chromadb_store):
        """Test vector addition"""
        chromadb_store.create_collection("test_collection", dimension=768)

        vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        ids = ["doc1", "doc2", "doc3"]
        documents = ["Document 1", "Document 2", "Document 3"]
        metadatas = [{"type": "test"} for _ in ids]

        result = chromadb_store.add_vectors(
            collection_name="test_collection",
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents
        )
        assert result is True

        # Verify vector count
        count = chromadb_store.count_vectors("test_collection")
        assert count == 3

    def test_search_vectors(self, chromadb_store):
        """Test vector similarity search"""
        # Setup test data
        chromadb_store.create_collection("test_collection", dimension=768)

        # Add similar and dissimilar vectors
        vectors = [
            [0.9] * 768,  # Very similar to query
            [0.1] * 768,  # Dissimilar to query
            [0.8] * 768   # Somewhat similar to query
        ]
        ids = ["similar", "dissimilar", "somewhat"]
        documents = ["Similar doc", "Different doc", "Somewhat similar doc"]
        metadatas = [{"similarity": "high"}, {"similarity": "low"}, {"similarity": "medium"}]

        chromadb_store.add_vectors("test_collection", ids, vectors, metadatas, documents)

        # Search with similar query vector
        query_vector = [0.85] * 768
        results = chromadb_store.search_vectors(
            collection_name="test_collection",
            query_embedding=query_vector,
            limit=2
        )

        assert len(results) == 2
        assert results[0].id == "similar"  # Most similar should be first
        assert results[0].similarity_score > results[1].similarity_score

    def test_delete_vectors(self, chromadb_store):
        """Test vector deletion"""
        # Setup test data
        chromadb_store.create_collection("test_collection", dimension=768)

        vectors = [[0.1] * 768, [0.2] * 768]
        ids = ["doc1", "doc2"]
        documents = ["Document 1", "Document 2"]
        metadatas = [{"type": "test"} for _ in ids]

        chromadb_store.add_vectors("test_collection", ids, vectors, metadatas, documents)

        # Delete one vector
        result = chromadb_store.delete_vectors("test_collection", ["doc1"])
        assert result is True

        # Verify deletion
        count = chromadb_store.count_vectors("test_collection")
        assert count == 1

        # Verify remaining vector
        results = chromadb_store.get_vectors("test_collection", ["doc2"])
        assert len(results) == 1
        assert results[0].id == "doc2"
```

#### Vector Store Interface Compliance Tests
```python
class TestVectorStoreCompliance:
    """Test that all vector store implementations comply with interface"""

    @pytest.fixture(params=["chromadb", "qdrant", "faiss"])
    def vector_store(self, request, tmp_path):
        """Parametrized fixture for all vector store implementations"""
        if request.param == "chromadb":
            return ChromaDBStore(persist_directory=str(tmp_path / "chromadb"))
        elif request.param == "qdrant":
            return QdrantStore(path=str(tmp_path / "qdrant"))
        elif request.param == "faiss":
            return FAISSStore(index_path=str(tmp_path / "faiss"))

    def test_interface_compliance(self, vector_store):
        """Test that all implementations satisfy the VectorStore interface"""
        assert hasattr(vector_store, 'create_collection')
        assert hasattr(vector_store, 'add_vectors')
        assert hasattr(vector_store, 'search_vectors')
        assert hasattr(vector_store, 'delete_vectors')
        assert hasattr(vector_store, 'count_vectors')
        assert hasattr(vector_store, 'list_collections')

    def test_basic_workflow(self, vector_store):
        """Test basic CRUD workflow works for all implementations"""
        # Create collection
        result = vector_store.create_collection("test", dimension=768)
        assert result is True

        # Add vectors
        vectors = [[0.1] * 768]
        result = vector_store.add_vectors(
            "test", ["doc1"], vectors, [{"test": True}], ["Test document"]
        )
        assert result is True

        # Search vectors
        results = vector_store.search_vectors("test", [0.1] * 768, limit=1)
        assert len(results) == 1
        assert results[0].id == "doc1"

        # Delete vectors
        result = vector_store.delete_vectors("test", ["doc1"])
        assert result is True

        # Verify deletion
        count = vector_store.count_vectors("test")
        assert count == 0
```

### 2. Embedding Service Tests

```python
class TestEmbeddingService:
    """Test Ollama-based embedding service"""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing"""
        config = EmbeddingConfig(
            model_name="nomic-embed-text",
            ollama_base_url="http://localhost:11434",
            timeout_seconds=30
        )
        return EmbeddingService(config)

    @pytest.mark.integration
    def test_generate_embedding(self, embedding_service):
        """Test single embedding generation"""
        text = "This is a test document for embedding generation."

        embedding = embedding_service.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Test that dimension is consistent
        embedding2 = embedding_service.generate_embedding("Different text")
        assert len(embedding) == len(embedding2)

    def test_empty_text_handling(self, embedding_service):
        """Test handling of empty or whitespace-only text"""
        with pytest.raises(ValueError):
            embedding_service.generate_embedding("")

        with pytest.raises(ValueError):
            embedding_service.generate_embedding("   ")

    @pytest.mark.integration
    def test_batch_embedding_generation(self, embedding_service):
        """Test batch embedding generation"""
        texts = [
            "First test document",
            "Second test document",
            "Third test document"
        ]

        embeddings = embedding_service.batch_generate_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)

        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1

    def test_similarity_calculation(self, embedding_service):
        """Test similarity calculation between embeddings"""
        # Create test embeddings
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Identical
        embedding3 = [0.9, 0.8, 0.7, 0.6, 0.5]  # Different

        # Test identical embeddings
        similarity_identical = embedding_service.calculate_similarity(
            embedding1, embedding2, metric="cosine"
        )
        assert similarity_identical == pytest.approx(1.0, abs=1e-6)

        # Test different embeddings
        similarity_different = embedding_service.calculate_similarity(
            embedding1, embedding3, metric="cosine"
        )
        assert 0.0 <= similarity_different <= 1.0
        assert similarity_different < similarity_identical

    def test_embedding_normalization(self, embedding_service):
        """Test that embeddings are properly normalized"""
        import numpy as np

        text = "Test document for normalization check"
        embedding = embedding_service.generate_embedding(text)

        # Check that embedding is normalized (unit vector)
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_embedding_consistency(self, embedding_service):
        """Test that same text produces consistent embeddings"""
        text = "Consistency test document"

        embedding1 = embedding_service.generate_embedding(text)
        embedding2 = embedding_service.generate_embedding(text)

        # Embeddings should be identical for same text
        similarity = embedding_service.calculate_similarity(
            embedding1, embedding2, metric="cosine"
        )
        assert similarity == pytest.approx(1.0, abs=1e-6)
```

### 3. Knowledge Manager Tests

```python
class TestKnowledgeManager:
    """Test core knowledge management functionality"""

    @pytest.fixture
    def knowledge_manager(self, tmp_path):
        """Create knowledge manager with test dependencies"""
        # Use in-memory/temporary storage for testing
        vector_store = ChromaDBStore(persist_directory=str(tmp_path / "chromadb"))
        embedding_service = MockEmbeddingService()  # Mock for fast tests
        metadata_db_path = str(tmp_path / "metadata.db")

        return KnowledgeManager(vector_store, embedding_service, metadata_db_path)

    def test_create_agent_collection(self, knowledge_manager):
        """Test agent collection creation"""
        agent_id = "test-agent-001"

        result = knowledge_manager.create_agent_collection(
            agent_id=agent_id,
            agent_name="Test Agent",
            agent_type="test"
        )

        assert result is True

        # Verify collection exists in vector store
        collections = knowledge_manager.vector_store.list_collections()
        assert f"agent-{agent_id}" in collections

    def test_store_knowledge(self, knowledge_manager):
        """Test knowledge storage"""
        agent_id = "test-agent-001"
        knowledge_manager.create_agent_collection(agent_id, "Test Agent", "test")

        knowledge_id = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content="Test knowledge content for storage",
            metadata={"topic": "testing", "difficulty": "easy"},
            tags=["test", "storage"],
            source="test_case"
        )

        assert knowledge_id is not None
        assert isinstance(knowledge_id, str)

        # Verify storage in vector database
        count = knowledge_manager.vector_store.count_vectors(f"agent-{agent_id}")
        assert count == 1

    def test_load_knowledge_by_query(self, knowledge_manager):
        """Test knowledge retrieval by semantic query"""
        agent_id = "test-agent-001"
        knowledge_manager.create_agent_collection(agent_id, "Test Agent", "test")

        # Store test knowledge
        test_knowledge = [
            ("Python is a programming language", ["python", "programming"]),
            ("Machine learning uses algorithms", ["ml", "algorithms"]),
            ("Databases store information", ["database", "storage"])
        ]

        stored_ids = []
        for content, tags in test_knowledge:
            knowledge_id = knowledge_manager.store_knowledge(
                agent_id=agent_id,
                content=content,
                tags=tags,
                source="test"
            )
            stored_ids.append(knowledge_id)

        # Query for programming-related knowledge
        results = knowledge_manager.load_knowledge(
            agent_id=agent_id,
            query="programming language syntax",
            limit=2,
            similarity_threshold=0.5
        )

        assert len(results) > 0
        assert all(isinstance(entry, KnowledgeEntry) for entry in results)
        assert all(entry.agent_id == agent_id for entry in results)

        # Results should be sorted by relevance score
        if len(results) > 1:
            scores = [entry.relevance_score for entry in results]
            assert scores == sorted(scores, reverse=True)

    def test_agent_isolation(self, knowledge_manager):
        """Test that agents cannot access each other's knowledge"""
        # Create two agents with separate knowledge
        agent1_id = "agent-001"
        agent2_id = "agent-002"

        knowledge_manager.create_agent_collection(agent1_id, "Agent 1", "test")
        knowledge_manager.create_agent_collection(agent2_id, "Agent 2", "test")

        # Store knowledge for agent 1
        agent1_knowledge_id = knowledge_manager.store_knowledge(
            agent_id=agent1_id,
            content="Agent 1 exclusive knowledge",
            tags=["agent1", "exclusive"]
        )

        # Store knowledge for agent 2
        agent2_knowledge_id = knowledge_manager.store_knowledge(
            agent_id=agent2_id,
            content="Agent 2 exclusive knowledge",
            tags=["agent2", "exclusive"]
        )

        # Agent 1 should only see its own knowledge
        agent1_results = knowledge_manager.load_knowledge(
            agent_id=agent1_id,
            query="exclusive knowledge",
            limit=10
        )

        assert len(agent1_results) == 1
        assert agent1_results[0].id == agent1_knowledge_id
        assert agent1_results[0].agent_id == agent1_id

        # Agent 2 should only see its own knowledge
        agent2_results = knowledge_manager.load_knowledge(
            agent_id=agent2_id,
            query="exclusive knowledge",
            limit=10
        )

        assert len(agent2_results) == 1
        assert agent2_results[0].id == agent2_knowledge_id
        assert agent2_results[0].agent_id == agent2_id

    def test_duplicate_content_handling(self, knowledge_manager):
        """Test handling of duplicate content"""
        agent_id = "test-agent-001"
        knowledge_manager.create_agent_collection(agent_id, "Test Agent", "test")

        content = "This is duplicate test content"

        # Store same content twice
        knowledge_id1 = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content=content,
            tags=["test"]
        )

        knowledge_id2 = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content=content,
            tags=["test"]
        )

        # Should return same ID for duplicate content
        assert knowledge_id1 == knowledge_id2

        # Should only have one entry in vector database
        count = knowledge_manager.vector_store.count_vectors(f"agent-{agent_id}")
        assert count == 1
```

### 4. RAG Integration Tests

```python
class TestRAGIntegration:
    """Test end-to-end RAG functionality"""

    @pytest.fixture
    def rag_system(self, tmp_path):
        """Setup complete RAG system for integration testing"""
        # Initialize components
        vector_store = ChromaDBStore(persist_directory=str(tmp_path / "chromadb"))
        embedding_service = EmbeddingService()  # Real embedding service
        knowledge_manager = KnowledgeManager(
            vector_store,
            embedding_service,
            str(tmp_path / "metadata.db")
        )

        # Create test agent
        agent_id = "integration-test-agent"
        knowledge_manager.create_agent_collection(
            agent_id=agent_id,
            agent_name="Integration Test Agent",
            agent_type="test"
        )

        return {
            'knowledge_manager': knowledge_manager,
            'agent_id': agent_id,
            'vector_store': vector_store,
            'embedding_service': embedding_service
        }

    @pytest.mark.integration
    def test_document_learning_workflow(self, rag_system):
        """Test complete document learning and retrieval workflow"""
        km = rag_system['knowledge_manager']
        agent_id = rag_system['agent_id']

        # Sample document content
        document_content = """
        Calculus is a branch of mathematics focused on limits, functions,
        derivatives, integrals, and infinite series. The derivative of a
        function represents the rate of change. For example, the derivative
        of x² is 2x. Integration is the reverse process of differentiation.
        The fundamental theorem of calculus connects derivatives and integrals.
        """

        # Process document (chunk and store)
        chunks = km.chunker.chunk_text(document_content, chunk_size=200)
        stored_ids = []

        for i, chunk in enumerate(chunks):
            knowledge_id = km.store_knowledge(
                agent_id=agent_id,
                content=chunk,
                metadata={
                    "document": "calculus_basics",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                tags=["calculus", "mathematics", "education"],
                source="document_learning"
            )
            stored_ids.append(knowledge_id)

        assert len(stored_ids) == len(chunks)

        # Test retrieval with relevant query
        results = km.load_knowledge(
            agent_id=agent_id,
            query="What is the derivative of x squared?",
            limit=3,
            similarity_threshold=0.6
        )

        assert len(results) > 0

        # Should retrieve relevant content about derivatives
        relevant_content = " ".join([r.content for r in results])
        assert "derivative" in relevant_content.lower()
        assert "2x" in relevant_content or "x²" in relevant_content

    @pytest.mark.integration
    def test_rag_agent_query_processing(self, rag_system):
        """Test RAG-enhanced agent query processing"""
        km = rag_system['knowledge_manager']
        agent_id = rag_system['agent_id']

        # Store relevant knowledge
        knowledge_entries = [
            "Python lists are ordered collections of items",
            "List comprehensions provide a concise way to create lists",
            "The append() method adds items to the end of a list",
            "Lists can contain different data types in Python"
        ]

        for content in knowledge_entries:
            km.store_knowledge(
                agent_id=agent_id,
                content=content,
                tags=["python", "programming", "lists"],
                source="knowledge_base"
            )

        # Create RAG-enabled agent
        rag_agent = RAGAgent(
            agent_id=agent_id,
            name="Test RAG Agent",
            instructions="You are a Python programming assistant.",
            model="llama3.2",
            knowledge_manager=km
        )

        # Process query with RAG
        query = "How do I add items to a Python list?"

        # Mock the agent processing (since we don't have real LLM in tests)
        context_knowledge = km.load_knowledge(
            agent_id=agent_id,
            query=query,
            limit=3,
            similarity_threshold=0.7
        )

        assert len(context_knowledge) > 0

        # Verify relevant knowledge was retrieved
        relevant_content = " ".join([k.content for k in context_knowledge])
        assert "append" in relevant_content.lower()
        assert "list" in relevant_content.lower()

    @pytest.mark.integration
    def test_multi_agent_isolation(self, rag_system):
        """Test that multiple agents maintain knowledge isolation"""
        km = rag_system['knowledge_manager']

        # Create multiple agents
        agents = [
            ("math-agent", "Mathematics Tutor", "math"),
            ("code-agent", "Programming Assistant", "coding"),
            ("write-agent", "Creative Writer", "writing")
        ]

        agent_knowledge = {}

        for agent_id, name, agent_type in agents:
            # Create agent collection
            km.create_agent_collection(agent_id, name, agent_type)

            # Store domain-specific knowledge
            if agent_type == "math":
                knowledge = [
                    "The derivative of sin(x) is cos(x)",
                    "Integration by parts: ∫u dv = uv - ∫v du",
                    "The chain rule for derivatives"
                ]
            elif agent_type == "coding":
                knowledge = [
                    "Python uses indentation for code blocks",
                    "Functions are defined with the def keyword",
                    "Classes use the class keyword"
                ]
            else:  # writing
                knowledge = [
                    "Character development is crucial for storytelling",
                    "Plot structure includes exposition, rising action, climax",
                    "Dialogue should sound natural and advance the plot"
                ]

            agent_knowledge[agent_id] = []
            for content in knowledge:
                knowledge_id = km.store_knowledge(
                    agent_id=agent_id,
                    content=content,
                    tags=[agent_type],
                    source="domain_knowledge"
                )
                agent_knowledge[agent_id].append(knowledge_id)

        # Test cross-agent isolation
        for agent_id, name, agent_type in agents:
            # Query for domain-specific knowledge
            if agent_type == "math":
                query = "derivatives and calculus"
            elif agent_type == "coding":
                query = "Python programming syntax"
            else:
                query = "story writing and characters"

            results = km.load_knowledge(
                agent_id=agent_id,
                query=query,
                limit=10
            )

            # Should only return knowledge for this agent
            assert all(r.agent_id == agent_id for r in results)
            assert len(results) > 0

            # Verify content is domain-specific
            content_text = " ".join([r.content for r in results])
            if agent_type == "math":
                assert any(word in content_text.lower()
                          for word in ["derivative", "integral", "calculus"])
            elif agent_type == "coding":
                assert any(word in content_text.lower()
                          for word in ["python", "function", "class"])
            else:
                assert any(word in content_text.lower()
                          for word in ["character", "plot", "story"])
```

This comprehensive testing strategy covers all critical aspects of the RAG and Vector Database system, ensuring reliability, performance, and correctness across all components and integration points.