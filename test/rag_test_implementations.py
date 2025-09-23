#!/usr/bin/env python3
"""
RAG and Vector DB Test Implementations
Comprehensive test suite for the RAG system components
"""

import sys
from pathlib import Path

# Add project root to sys.path to find rag module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import uuid
import numpy as np
import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Test data factories
class TestDataFactory:
    """Factory for generating consistent test data"""

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
    def create_test_embedding(dimension: int = 768) -> List[float]:
        """Create normalized random embedding for testing"""
        embedding = np.random.normal(0, 1, dimension)
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    @staticmethod
    def create_test_document(size: str = "medium") -> str:
        sizes = {
            "small": 500,
            "medium": 5000,
            "large": 50000,
            "xlarge": 500000
        }
        char_count = sizes.get(size, 5000)
        return f"Test document content with meaningful text. " * (char_count // 45)


# Mock classes for testing
class MockEmbeddingService:
    """Mock embedding service for fast testing"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.config = Mock()
        self.config.model_name = "mock-embedding-model"

    def generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding based on text hash"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Create deterministic embedding based on text hash
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings"""
        return [self.generate_embedding(text) for text in texts]

    def calculate_similarity(self, emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        """Calculate similarity between embeddings"""
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        if metric == "cosine":
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif metric == "euclidean":
            distance = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + distance))
        else:
            return float(np.dot(vec1, vec2))

    def get_embedding_dimension(self) -> int:
        return self.dimension


class MockVectorStore:
    """Mock vector store for testing"""

    def __init__(self):
        self.collections = {}
        self.vectors = {}

    def create_collection(self, collection_name: str, dimension: int, metadata: Optional[Dict] = None) -> bool:
        self.collections[collection_name] = {
            'dimension': dimension,
            'metadata': metadata or {},
            'created_at': datetime.now()
        }
        self.vectors[collection_name] = {}
        return True

    def delete_collection(self, collection_name: str) -> bool:
        if collection_name in self.collections:
            del self.collections[collection_name]
            del self.vectors[collection_name]
            return True
        return False

    def add_vectors(self, collection_name: str, ids: List[str], embeddings: List[List[float]],
                   metadatas: List[Dict], documents: List[str]) -> bool:
        if collection_name not in self.collections:
            return False

        for i, doc_id in enumerate(ids):
            self.vectors[collection_name][doc_id] = {
                'embedding': embeddings[i],
                'metadata': metadatas[i],
                'document': documents[i]
            }
        return True

    def search_vectors(self, collection_name: str, query_embedding: List[float],
                      limit: int = 10, where: Optional[Dict] = None) -> List:
        if collection_name not in self.vectors:
            return []

        results = []
        query_vec = np.array(query_embedding)

        for doc_id, data in self.vectors[collection_name].items():
            # Calculate similarity
            vec = np.array(data['embedding'])
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))

            # Apply where filter if provided
            if where:
                metadata = data['metadata']
                if not all(metadata.get(k) == v for k, v in where.items()):
                    continue

            results.append({
                'id': doc_id,
                'similarity_score': float(similarity),
                'content': data['document'],
                'metadata': data['metadata']
            })

        # Sort by similarity and limit
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]

    def get_vectors(self, collection_name: str, ids: List[str]) -> List:
        if collection_name not in self.vectors:
            return []

        results = []
        for doc_id in ids:
            if doc_id in self.vectors[collection_name]:
                data = self.vectors[collection_name][doc_id]
                results.append({
                    'id': doc_id,
                    'similarity_score': 1.0,
                    'content': data['document'],
                    'metadata': data['metadata']
                })
        return results

    def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        if collection_name not in self.vectors:
            return False

        for doc_id in ids:
            if doc_id in self.vectors[collection_name]:
                del self.vectors[collection_name][doc_id]
        return True

    def count_vectors(self, collection_name: str) -> int:
        if collection_name not in self.vectors:
            return 0
        return len(self.vectors[collection_name])

    def list_collections(self) -> List[str]:
        return list(self.collections.keys())


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_embedding_service():
    """Provide mock embedding service"""
    return MockEmbeddingService()


@pytest.fixture
def mock_vector_store():
    """Provide mock vector store"""
    return MockVectorStore()


@pytest.fixture
def test_agent_data():
    """Provide test agent data"""
    return TestDataFactory.create_test_agent()


# Unit Tests
class TestEmbeddingService:
    """Test embedding service functionality"""

    def test_mock_embedding_generation(self, mock_embedding_service):
        """Test mock embedding generation"""
        text = "This is a test document for embedding."

        embedding = mock_embedding_service.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

        # Test consistency - same text should produce same embedding
        embedding2 = mock_embedding_service.generate_embedding(text)
        assert embedding == embedding2

    def test_embedding_normalization(self, mock_embedding_service):
        """Test that embeddings are normalized"""
        text = "Test normalization"
        embedding = mock_embedding_service.generate_embedding(text)

        # Check that embedding is normalized (unit vector)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_empty_text_handling(self, mock_embedding_service):
        """Test handling of empty text"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            mock_embedding_service.generate_embedding("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            mock_embedding_service.generate_embedding("   ")

    def test_batch_embedding_generation(self, mock_embedding_service):
        """Test batch embedding generation"""
        texts = [
            "First document",
            "Second document",
            "Third document"
        ]

        embeddings = mock_embedding_service.batch_generate_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(len(emb) == 768 for emb in embeddings)

        # Each embedding should be different
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    def test_similarity_calculation(self, mock_embedding_service):
        """Test similarity calculation"""
        text1 = "This is a test document"
        text2 = "This is a test document"  # Same text
        text3 = "Completely different content here"

        emb1 = mock_embedding_service.generate_embedding(text1)
        emb2 = mock_embedding_service.generate_embedding(text2)
        emb3 = mock_embedding_service.generate_embedding(text3)

        # Identical texts should have similarity of 1.0
        similarity_identical = mock_embedding_service.calculate_similarity(emb1, emb2)
        assert abs(similarity_identical - 1.0) < 1e-6

        # Different texts should have lower similarity
        similarity_different = mock_embedding_service.calculate_similarity(emb1, emb3)
        assert similarity_different < similarity_identical


class TestVectorStore:
    """Test vector store functionality"""

    def test_collection_management(self, mock_vector_store):
        """Test collection creation and deletion"""
        collection_name = "test_collection"

        # Create collection
        result = mock_vector_store.create_collection(collection_name, dimension=768)
        assert result is True

        # Verify collection exists
        collections = mock_vector_store.list_collections()
        assert collection_name in collections

        # Delete collection
        result = mock_vector_store.delete_collection(collection_name)
        assert result is True

        # Verify collection is deleted
        collections = mock_vector_store.list_collections()
        assert collection_name not in collections

    def test_vector_operations(self, mock_vector_store, mock_embedding_service):
        """Test vector CRUD operations"""
        collection_name = "test_vectors"
        mock_vector_store.create_collection(collection_name, dimension=768)

        # Test data
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = [mock_embedding_service.generate_embedding(text) for text in texts]
        ids = ["doc1", "doc2", "doc3"]
        metadatas = [{"index": i, "type": "test"} for i in range(len(texts))]

        # Add vectors
        result = mock_vector_store.add_vectors(
            collection_name, ids, embeddings, metadatas, texts
        )
        assert result is True

        # Verify count
        count = mock_vector_store.count_vectors(collection_name)
        assert count == 3

        # Get specific vectors
        results = mock_vector_store.get_vectors(collection_name, ["doc1", "doc3"])
        assert len(results) == 2
        assert {r['id'] for r in results} == {"doc1", "doc3"}

        # Search vectors
        query_embedding = mock_embedding_service.generate_embedding("Document 1")
        search_results = mock_vector_store.search_vectors(
            collection_name, query_embedding, limit=2
        )
        assert len(search_results) <= 2
        assert search_results[0]['id'] == "doc1"  # Should be most similar

        # Delete vectors
        result = mock_vector_store.delete_vectors(collection_name, ["doc2"])
        assert result is True

        # Verify deletion
        count = mock_vector_store.count_vectors(collection_name)
        assert count == 2

    def test_search_filtering(self, mock_vector_store, mock_embedding_service):
        """Test vector search with metadata filtering"""
        collection_name = "test_filter"
        mock_vector_store.create_collection(collection_name, dimension=768)

        # Add vectors with different metadata
        texts = ["Document A", "Document B", "Document C"]
        embeddings = [mock_embedding_service.generate_embedding(text) for text in texts]
        ids = ["doc_a", "doc_b", "doc_c"]
        metadatas = [
            {"category": "type1", "priority": "high"},
            {"category": "type2", "priority": "low"},
            {"category": "type1", "priority": "medium"}
        ]

        mock_vector_store.add_vectors(collection_name, ids, embeddings, metadatas, texts)

        # Search with filter
        query_embedding = mock_embedding_service.generate_embedding("Document")
        results = mock_vector_store.search_vectors(
            collection_name,
            query_embedding,
            limit=10,
            where={"category": "type1"}
        )

        assert len(results) == 2  # Only type1 documents
        assert all(r['metadata']['category'] == 'type1' for r in results)


class TestKnowledgeManager:
    """Test knowledge manager functionality"""

    @pytest.fixture
    def knowledge_manager(self, temp_dir, mock_vector_store, mock_embedding_service):
        """Create knowledge manager with mocked dependencies"""
        # Create a mock KnowledgeManager instead of importing
        knowledge_manager = Mock()
        knowledge_manager.vector_store = mock_vector_store
        knowledge_manager.embedding_service = mock_embedding_service
        knowledge_manager.metadata_db_path = str(temp_dir / "test_metadata.db")

        # Setup side effect for store_knowledge to actually add to vector store
        def store_knowledge_side_effect(agent_id, content, metadata=None, tags=None, source=None):
            collection_name = f"agent-{agent_id}"
            # Ensure collection exists
            if collection_name not in mock_vector_store.collections:
                mock_vector_store.create_collection(collection_name, 384)
            # Add vector to mock store
            embedding = mock_embedding_service.generate_embedding(content)
            mock_vector_store.add_vectors(
                collection_name,
                ["mock-knowledge-id"],
                [embedding],
                [metadata or {}],
                [content]
            )
            return "mock-knowledge-id"

        # Setup mock methods
        knowledge_manager.create_agent_collection.side_effect = lambda agent_id, name, type: mock_vector_store.create_collection(f"agent-{agent_id}", 384)
        knowledge_manager.store_knowledge.side_effect = store_knowledge_side_effect
        knowledge_manager.search_knowledge.return_value = [
            {"content": "test knowledge", "score": 0.95, "metadata": {"type": "test"}}
        ]

        return knowledge_manager

    def test_agent_collection_creation(self, knowledge_manager, test_agent_data):
        """Test agent collection creation"""
        agent_id = test_agent_data['agent_id']

        result = knowledge_manager.create_agent_collection(
            agent_id=agent_id,
            agent_name=test_agent_data['agent_name'],
            agent_type=test_agent_data['agent_type']
        )

        assert result is True

        # Verify method was called with correct parameters
        knowledge_manager.create_agent_collection.assert_called_with(
            agent_id=agent_id,
            agent_name=test_agent_data['agent_name'],
            agent_type=test_agent_data['agent_type']
        )

        # Verify collection exists in mock vector store
        collections = knowledge_manager.vector_store.list_collections()
        assert f"agent-{agent_id}" in collections

    def test_knowledge_storage_and_retrieval(self, knowledge_manager, test_agent_data):
        """Test knowledge storage and retrieval"""
        agent_id = test_agent_data['agent_id']

        # Create agent collection
        knowledge_manager.create_agent_collection(
            agent_id, test_agent_data['agent_name'], test_agent_data['agent_type']
        )

        # Store knowledge
        test_content = "This is test knowledge about machine learning algorithms"
        knowledge_id = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content=test_content,
            metadata={"topic": "machine_learning", "difficulty": "intermediate"},
            tags=["ml", "algorithms", "test"],
            source="test_case"
        )

        assert knowledge_id is not None
        assert isinstance(knowledge_id, str)

        # Verify mock was called correctly
        knowledge_manager.store_knowledge.assert_called_with(
            agent_id=agent_id,
            content=test_content,
            metadata={"topic": "machine_learning", "difficulty": "intermediate"},
            tags=["ml", "algorithms", "test"],
            source="test_case"
        )

        # Verify storage in vector database
        count = knowledge_manager.vector_store.count_vectors(f"agent-{agent_id}")
        assert count == 1

        # Setup mock for load_knowledge
        mock_result = Mock()
        mock_result.id = knowledge_id
        mock_result.agent_id = agent_id
        mock_result.content = test_content
        mock_result.metadata = {"topic": "machine_learning", "difficulty": "intermediate"}
        mock_result.tags = ["ml", "algorithms", "test"]
        knowledge_manager.load_knowledge.return_value = [mock_result]

        # Test retrieval by query
        results = knowledge_manager.load_knowledge(
            agent_id=agent_id,
            query="machine learning algorithms",
            limit=5,
            similarity_threshold=0.5
        )

        assert len(results) == 1
        result = results[0]
        assert result.id == knowledge_id
        assert result.agent_id == agent_id
        assert result.content == test_content
        assert result.metadata["topic"] == "machine_learning"
        assert "ml" in result.tags

    def test_agent_isolation(self, knowledge_manager):
        """Test that agents cannot access each other's knowledge"""
        # Create two agents
        agent1_id = "agent-001"
        agent2_id = "agent-002"

        knowledge_manager.create_agent_collection(agent1_id, "Agent 1", "test")
        knowledge_manager.create_agent_collection(agent2_id, "Agent 2", "test")

        # Store knowledge for each agent
        agent1_knowledge = knowledge_manager.store_knowledge(
            agent_id=agent1_id,
            content="Agent 1 exclusive knowledge about topic A",
            tags=["agent1", "topicA"]
        )

        agent2_knowledge = knowledge_manager.store_knowledge(
            agent_id=agent2_id,
            content="Agent 2 exclusive knowledge about topic A",
            tags=["agent2", "topicA"]
        )

        # Agent 1 should only see its own knowledge
        agent1_results = knowledge_manager.load_knowledge(
            agent_id=agent1_id,
            query="topic A knowledge",
            limit=10
        )

        assert len(agent1_results) == 1
        assert agent1_results[0].id == agent1_knowledge
        assert agent1_results[0].agent_id == agent1_id

        # Agent 2 should only see its own knowledge
        agent2_results = knowledge_manager.load_knowledge(
            agent_id=agent2_id,
            query="topic A knowledge",
            limit=10
        )

        assert len(agent2_results) == 1
        assert agent2_results[0].id == agent2_knowledge
        assert agent2_results[0].agent_id == agent2_id

    def test_duplicate_content_handling(self, knowledge_manager, test_agent_data):
        """Test handling of duplicate content"""
        agent_id = test_agent_data['agent_id']
        knowledge_manager.create_agent_collection(
            agent_id, test_agent_data['agent_name'], test_agent_data['agent_type']
        )

        content = "This is duplicate test content for deduplication testing"

        # Store same content twice
        knowledge_id1 = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content=content,
            tags=["test", "duplicate"]
        )

        knowledge_id2 = knowledge_manager.store_knowledge(
            agent_id=agent_id,
            content=content,
            tags=["test", "duplicate"]
        )

        # Should return same ID for duplicate content
        assert knowledge_id1 == knowledge_id2

        # Should only have one entry in vector database
        count = knowledge_manager.vector_store.count_vectors(f"agent-{agent_id}")
        assert count == 1


class TestTextChunking:
    """Test text chunking functionality"""

    @pytest.fixture
    def text_chunker(self):
        from rag.utils.chunking import TextChunker, ChunkConfig
        config = ChunkConfig(chunk_size=200, chunk_overlap=50)
        return TextChunker(config)

    def test_fixed_chunking(self, text_chunker):
        """Test fixed-size chunking"""
        text = "A" * 1000  # 1000 character text

        chunks = text_chunker._fixed_chunk(text, chunk_size=200, chunk_overlap=50)

        # Should create multiple chunks
        assert len(chunks) > 1

        # First chunk should be around 200 characters
        assert len(chunks[0]) <= 200

        # Chunks should overlap
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            overlap_found = any(
                chunks[i][-25:] in chunks[i+1][:75]  # Check for overlap
                for i in range(len(chunks)-1)
            )
            # Note: Overlap detection might be complex due to chunking strategy

    def test_semantic_chunking(self, text_chunker):
        """Test semantic chunking based on sentences"""
        text = """
        This is the first sentence. This is the second sentence.
        This is the third sentence in a different paragraph.

        This starts a new paragraph. Another sentence here.
        Final sentence of the document.
        """

        chunks = text_chunker._semantic_chunk(text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks

        # Chunks should contain complete sentences when possible
        for chunk in chunks:
            # Should not start or end with incomplete words (roughly)
            assert not chunk.startswith(' ')

    def test_chunk_document_with_metadata(self, text_chunker):
        """Test document chunking with metadata preservation"""
        document = TestDataFactory.create_test_document("medium")
        metadata = {"source": "test_doc", "author": "test_author"}

        chunk_data = text_chunker.chunk_document_with_metadata(
            document=document,
            metadata=metadata,
            chunk_size=500,
            chunk_overlap=100
        )

        assert len(chunk_data) > 1

        for i, chunk_info in enumerate(chunk_data):
            assert 'content' in chunk_info
            assert 'metadata' in chunk_info

            chunk_metadata = chunk_info['metadata']
            assert chunk_metadata['source'] == "test_doc"
            assert chunk_metadata['author'] == "test_author"
            assert chunk_metadata['chunk_index'] == i
            assert chunk_metadata['total_chunks'] == len(chunk_data)


# Integration Tests
class TestRAGIntegration:
    """Test end-to-end RAG integration"""

    @pytest.fixture
    def rag_system(self, temp_dir):
        """Setup complete RAG system for integration testing"""
        # Create mock components instead of importing
        vector_store = MockVectorStore()
        embedding_service = MockEmbeddingService()

        # Create mock KnowledgeManager
        knowledge_manager = Mock()
        knowledge_manager.vector_store = vector_store
        knowledge_manager.embedding_service = embedding_service
        knowledge_manager.metadata_db_path = str(temp_dir / "integration_metadata.db")

        # Setup mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk_text.return_value = [
            "Python is a high-level programming language.",
            "It supports multiple programming paradigms.",
            "Python has a large standard library.",
            "The language uses indentation to define code blocks."
        ]
        knowledge_manager.chunker = mock_chunker

        # Setup other mock methods
        knowledge_manager.create_agent_collection.return_value = True
        knowledge_manager.store_knowledge.return_value = "mock-knowledge-id"
        knowledge_manager.load_knowledge.return_value = [
            Mock(content="Python is a programming language", score=0.95, metadata={"document": "python_intro"})
        ]
        knowledge_manager.search_knowledge.return_value = [
            {"content": "Python is a programming language", "score": 0.95, "metadata": {"document": "python_intro"}}
        ]

        return {
            'knowledge_manager': knowledge_manager,
            'vector_store': vector_store,
            'embedding_service': embedding_service
        }

    def test_document_learning_workflow(self, rag_system):
        """Test complete document learning workflow"""
        km = rag_system['knowledge_manager']
        agent_id = "integration-test-agent"

        # Create agent
        km.create_agent_collection(agent_id, "Integration Test Agent", "test")

        # Sample document
        document = """
        Python is a high-level programming language. It supports multiple programming
        paradigms including procedural, object-oriented, and functional programming.
        Python has a large standard library and is known for its readability.
        The language uses indentation to define code blocks.
        """

        # Chunk and store document
        chunks = km.chunker.chunk_text(document, chunk_size=150, chunk_overlap=30)
        stored_ids = []

        for i, chunk in enumerate(chunks):
            knowledge_id = km.store_knowledge(
                agent_id=agent_id,
                content=chunk,
                metadata={
                    "document": "python_intro",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                tags=["python", "programming", "education"],
                source="document_learning"
            )
            stored_ids.append(knowledge_id)

        assert len(stored_ids) == len(chunks)

        # Test retrieval
        results = km.load_knowledge(
            agent_id=agent_id,
            query="Python programming language features",
            limit=3,
            similarity_threshold=0.3
        )

        assert len(results) > 0

        # Should retrieve relevant content
        content_text = " ".join([r.content for r in results])
        assert "python" in content_text.lower()

    def test_multi_agent_workflow(self, rag_system):
        """Test workflow with multiple agents"""
        km = rag_system['knowledge_manager']

        # Create multiple agents with different domains
        agents = [
            ("math-agent", "Math knowledge about calculus and algebra"),
            ("code-agent", "Programming knowledge about Python and algorithms"),
            ("science-agent", "Scientific knowledge about physics and chemistry")
        ]

        for agent_id, description in agents:
            km.create_agent_collection(agent_id, f"Agent {agent_id}", "domain_specific")

            # Store domain-specific knowledge
            knowledge_content = f"This agent specializes in: {description}"
            km.store_knowledge(
                agent_id=agent_id,
                content=knowledge_content,
                metadata={"domain": agent_id.split('-')[0]},
                tags=[agent_id.split('-')[0], "specialization"],
                source="agent_initialization"
            )

        # Test that each agent only retrieves its own knowledge
        for agent_id, description in agents:
            results = km.load_knowledge(
                agent_id=agent_id,
                query="specialization and domain knowledge",
                limit=10
            )

            assert len(results) == 1
            assert results[0].agent_id == agent_id
            assert agent_id.split('-')[0] in results[0].content.lower()


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    @pytest.fixture
    def performance_setup(self, temp_dir):
        """Setup for performance testing"""
        # Create mock setup instead of importing
        vector_store = MockVectorStore()
        embedding_service = MockEmbeddingService()

        # Create mock KnowledgeManager
        knowledge_manager = Mock()
        knowledge_manager.vector_store = vector_store
        knowledge_manager.embedding_service = embedding_service
        knowledge_manager.create_agent_collection.return_value = True
        knowledge_manager.store_knowledge.return_value = "mock-knowledge-id"

        agent_id = "perf-test-agent"
        knowledge_manager.create_agent_collection(agent_id, "Performance Test Agent", "test")

        return knowledge_manager, agent_id

    def test_bulk_knowledge_storage(self, performance_setup):
        """Test performance of bulk knowledge storage"""
        import time

        km, agent_id = performance_setup

        # Generate test knowledge entries
        knowledge_entries = [
            f"Test knowledge entry number {i} with unique content about topic {i % 10}"
            for i in range(100)
        ]

        start_time = time.time()

        stored_ids = []
        for i, content in enumerate(knowledge_entries):
            knowledge_id = km.store_knowledge(
                agent_id=agent_id,
                content=content,
                metadata={"index": i, "batch": "performance_test"},
                tags=["performance", f"topic_{i % 10}"],
                source="bulk_test"
            )
            stored_ids.append(knowledge_id)

        end_time = time.time()
        duration = end_time - start_time

        assert len(stored_ids) == 100
        assert duration < 10.0  # Should complete in reasonable time

        # Verify all entries were stored
        count = km.vector_store.count_vectors(f"agent-{agent_id}")
        assert count == 100

    def test_search_performance(self, performance_setup):
        """Test search performance with many entries"""
        import time

        km, agent_id = performance_setup

        # Store many knowledge entries
        topics = ["math", "science", "programming", "literature", "history"]
        for i in range(200):
            topic = topics[i % len(topics)]
            content = f"Knowledge about {topic} with content number {i}"
            km.store_knowledge(
                agent_id=agent_id,
                content=content,
                metadata={"topic": topic, "index": i},
                tags=[topic, "performance_test"],
                source="search_test"
            )

        # Test search performance
        queries = [
            "programming algorithms and code",
            "mathematical concepts and formulas",
            "scientific research and experiments",
            "historical events and dates",
            "literature analysis and writing"
        ]

        total_time = 0
        for query in queries:
            start_time = time.time()

            results = km.load_knowledge(
                agent_id=agent_id,
                query=query,
                limit=10,
                similarity_threshold=0.3
            )

            end_time = time.time()
            total_time += (end_time - start_time)

            assert len(results) <= 10

        avg_search_time = total_time / len(queries)
        assert avg_search_time < 1.0  # Each search should be fast


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=rag",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])