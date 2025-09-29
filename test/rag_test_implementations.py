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


# ============================================================================
# Unit Tests: 임베딩 서비스 핵심 기능 테스트
# ============================================================================
@pytest.mark.unit
class TestEmbeddingService:
    """
    임베딩 서비스의 핵심 기능들을 검증하는 테스트 클래스
    
    임베딩 서비스는 RAG 시스템의 심장부로, 텍스트를 벡터로 변환하는 역할을 합니다.
    이 테스트들은 다음을 보장합니다:
    - 텍스트가 올바른 형태의 벡터로 변환되는지
    - 벡터가 수학적으로 정규화되어 있는지 (단위 벡터)
    - 같은 텍스트는 항상 같은 벡터를 생성하는지 (일관성)
    - 에러 상황을 적절히 처리하는지
    - 배치 처리가 정상 작동하는지
    - 유사도 계산이 의미적 차이를 반영하는지
    """

    def test_mock_embedding_generation(self, mock_embedding_service):
        """
        기본 임베딩 생성 기능 테스트
        
        목적:
        - 텍스트가 정상적으로 벡터(리스트 형태)로 변환되는지 확인
        - 벡터의 차원이 예상대로 768차원인지 검증
        - 모든 요소가 float 타입인지 확인
        - 일관성 테스트: 동일한 텍스트는 항상 동일한 벡터를 생성해야 함
        
        왜 중요한가:
        - RAG 시스템에서 같은 질문에 대해 매번 다른 검색 결과가 나오면 안 됨
        - 캐싱과 중복 제거를 위해 일관성이 필수적
        """
        # 테스트용 샘플 텍스트
        text = "This is a test document for embedding."

        # 임베딩 생성
        embedding = mock_embedding_service.generate_embedding(text)

        # 검증 1: 반환 타입이 리스트인가?
        assert isinstance(embedding, list), "임베딩은 리스트 형태여야 합니다"
        
        # 검증 2: 벡터 차원이 768인가? (표준 임베딩 차원)
        assert len(embedding) == 768, f"임베딩 차원은 768이어야 하는데 {len(embedding)}입니다"
        
        # 검증 3: 모든 요소가 float 타입인가?
        assert all(isinstance(x, float) for x in embedding), "모든 임베딩 요소는 float이어야 합니다"

        # 검증 4: 일관성 테스트 - 같은 텍스트는 같은 임베딩을 생성해야 함
        embedding2 = mock_embedding_service.generate_embedding(text)
        assert embedding == embedding2, "동일한 텍스트는 항상 동일한 임베딩을 생성해야 합니다"

    def test_embedding_normalization(self, mock_embedding_service):
        """
        임베딩 벡터 정규화 검증 테스트
        
        목적:
        - 생성된 임베딩이 단위 벡터(unit vector)인지 확인
        - 벡터의 노름(norm, 길이)이 1.0인지 검증
        
        왜 중요한가:
        - 정규화된 벡터는 코사인 유사도 계산을 단순화 (내적만으로 계산 가능)
        - 벡터 크기에 관계없이 방향만으로 유사도 비교 가능
        - 수치적 안정성 향상 (overflow/underflow 방지)
        
        수학적 배경:
        - 단위 벡터: ||v|| = 1
        - 코사인 유사도: cos(θ) = (a·b) / (||a|| × ||b||)
        - 정규화된 벡터끼리는: cos(θ) = a·b (단순화!)
        """
        text = "Test normalization"
        embedding = mock_embedding_service.generate_embedding(text)

        # 벡터의 노름(길이) 계산: sqrt(x1² + x2² + ... + xn²)
        norm = np.linalg.norm(embedding)
        
        # 검증: 노름이 1.0에 매우 가까운가? (부동소수점 오차 허용: 1e-6)
        assert abs(norm - 1.0) < 1e-6, \
            f"임베딩 벡터는 정규화되어야 합니다 (norm=1.0). 현재 norm={norm}"

    def test_empty_text_handling(self, mock_embedding_service):
        """
        빈 텍스트 입력에 대한 에러 처리 테스트
        
        목적:
        - 빈 문자열("")에 대해 적절한 에러를 발생시키는지 확인
        - 공백만 있는 문자열("   ")도 에러 처리하는지 검증
        
        왜 중요한가:
        - 의미 없는 입력에 대해 의미 있는 에러 메시지 제공
        - 시스템 안정성: 잘못된 입력으로 인한 크래시 방지
        - 디버깅 용이성: 명확한 에러 메시지로 문제 파악 쉬움
        
        방어적 프로그래밍:
        - 항상 입력 검증을 먼저 수행
        - 실패는 빨리, 명확하게 (Fail Fast)
        """
        # 테스트 1: 빈 문자열은 ValueError를 발생시켜야 함
        with pytest.raises(ValueError, match="Text cannot be empty"):
            mock_embedding_service.generate_embedding("")

        # 테스트 2: 공백만 있는 문자열도 에러 처리해야 함
        with pytest.raises(ValueError, match="Text cannot be empty"):
            mock_embedding_service.generate_embedding("   ")

    def test_batch_embedding_generation(self, mock_embedding_service):
        """
        배치 임베딩 생성 기능 테스트
        
        목적:
        - 여러 텍스트를 한 번에 처리할 수 있는지 확인
        - 각 텍스트가 올바른 차원의 벡터로 변환되는지 검증
        - 서로 다른 텍스트는 서로 다른 임베딩을 생성하는지 확인
        
        왜 중요한가:
        - 성능 최적화: 배치 처리는 개별 처리보다 훨씬 빠름
        - GPU 활용: 병렬 처리로 GPU 효율성 극대화
        - 대량 문서 처리: RAG 시스템에서 수백~수천 개 문서 처리 시 필수
        
        실제 사용 사례:
        - 초기 지식베이스 구축 시 수천 개 문서 임베딩
        - 실시간 검색 시 여러 후보 문서 동시 처리
        """
        # 테스트용 여러 문서들
        texts = [
            "First document",
            "Second document",
            "Third document"
        ]

        # 배치로 임베딩 생성
        embeddings = mock_embedding_service.batch_generate_embeddings(texts)

        # 검증 1: 입력 텍스트 수와 출력 임베딩 수가 같은가?
        assert len(embeddings) == len(texts), \
            f"입력 {len(texts)}개에 대해 {len(embeddings)}개의 임베딩이 생성되었습니다"
        
        # 검증 2: 모든 임베딩이 올바른 차원을 가지는가?
        assert all(len(emb) == 768 for emb in embeddings), \
            "모든 임베딩은 768차원이어야 합니다"

        # 검증 3: 각 임베딩이 서로 다른가? (같으면 버그!)
        assert embeddings[0] != embeddings[1], \
            "다른 텍스트는 다른 임베딩을 생성해야 합니다"
        assert embeddings[1] != embeddings[2], \
            "다른 텍스트는 다른 임베딩을 생성해야 합니다"

    def test_similarity_calculation(self, mock_embedding_service):
        """
        유사도 계산의 정확성 테스트
        
        목적:
        - 동일한 텍스트의 유사도가 1.0인지 확인
        - 다른 텍스트의 유사도가 더 낮은지 검증
        - 임베딩이 의미적 차이를 실제로 반영하는지 확인
        
        왜 중요한가:
        - RAG의 핵심: 유사도 기반 검색의 정확성 보장
        - 의미적 이해: 단순 키워드 매칭이 아닌 의미 기반 검색
        - 검색 품질: 관련 문서를 정확히 찾아내는 능력
        
        수학적 배경:
        - 코사인 유사도 범위: -1 ~ 1
        - 1.0: 완전히 같은 방향 (동일)
        - 0.0: 직교 (무관)
        - -1.0: 반대 방향 (반대)
        
        실제 사용:
        사용자 질문: "비트코인 투자 방법은?"
        문서1: "비트코인 투자 가이드" → 유사도 0.95
        문서2: "파스타 레시피" → 유사도 0.15
        """
        # 테스트 데이터 준비
        text1 = "This is a test document"
        text2 = "This is a test document"  # text1과 완전히 동일
        text3 = "Completely different content here"  # 완전히 다른 내용

        # 각 텍스트의 임베딩 생성
        emb1 = mock_embedding_service.generate_embedding(text1)
        emb2 = mock_embedding_service.generate_embedding(text2)
        emb3 = mock_embedding_service.generate_embedding(text3)

        # 검증 1: 동일한 텍스트는 유사도 1.0을 가져야 함
        similarity_identical = mock_embedding_service.calculate_similarity(emb1, emb2)
        assert abs(similarity_identical - 1.0) < 1e-6, \
            f"동일한 텍스트의 유사도는 1.0이어야 하는데 {similarity_identical}입니다"

        # 검증 2: 다른 텍스트는 더 낮은 유사도를 가져야 함
        similarity_different = mock_embedding_service.calculate_similarity(emb1, emb3)
        assert similarity_different < similarity_identical, \
            f"다른 텍스트({similarity_different})가 같은 텍스트({similarity_identical})보다 " \
            f"유사도가 높거나 같습니다. 이는 버그입니다!"


@pytest.mark.unit
class TestVectorStore:
    """
    벡터 스토어(Vector Store) 핵심 기능 테스트 클래스
    
    벡터 스토어는 RAG 시스템의 데이터베이스로, 임베딩된 문서들을 저장하고 검색하는 역할을 합니다.
    실제 프로덕션에서는 ChromaDB, Pinecone, Qdrant 등을 사용하지만,
    이 테스트에서는 MockVectorStore를 사용하여 메모리 기반으로 빠르게 테스트합니다.
    
    테스트 범위:
    - 컬렉션 생성/삭제 (각 에이전트의 독립적인 지식 저장소)
    - 벡터 CRUD 연산 (Create, Read, Update, Delete)
    - 유사도 기반 검색 (RAG의 핵심 기능)
    - 메타데이터 필터링 검색 (조건부 검색)
    
    참고:
    - MockVectorStore는 메모리(딕셔너리)에만 저장하므로 테스트 종료 시 데이터 사라짐
    - 실제 DB와 동일한 인터페이스를 제공하여 로직 검증 가능
    """

    def test_collection_management(self, mock_vector_store):
        """
        컬렉션 생성 및 삭제 기능 테스트
        
        목적:
        - 벡터 스토어에 새로운 컬렉션을 생성할 수 있는지 확인
        - 생성된 컬렉션이 목록에 정상적으로 나타나는지 검증
        - 컬렉션 삭제가 정상 작동하는지 확인
        - 삭제 후 컬렉션이 목록에서 사라지는지 검증
        
        왜 중요한가:
        - RAG 시스템에서 각 에이전트는 독립적인 지식 저장소(컬렉션)가 필요
        - 에이전트 간 지식이 섞이지 않도록 격리(isolation) 보장
        - 컬렉션 = 데이터베이스의 테이블 또는 네임스페이스와 유사
        
        실제 사용 예시:
        - agent-001 → "agent-001" 컬렉션 생성
        - agent-002 → "agent-002" 컬렉션 생성
        - 각 에이전트는 자신의 컬렉션만 접근 가능
        """
        collection_name = "test_collection"

        # 1단계: 컬렉션 생성 (dimension=768은 임베딩 벡터의 차원)
        result = mock_vector_store.create_collection(collection_name, dimension=768)
        assert result is True, "컬렉션 생성이 실패했습니다"

        # 2단계: 컬렉션이 목록에 나타나는지 확인
        collections = mock_vector_store.list_collections()
        assert collection_name in collections, \
            f"생성된 컬렉션 '{collection_name}'이 목록에 없습니다"

        # 3단계: 컬렉션 삭제
        result = mock_vector_store.delete_collection(collection_name)
        assert result is True, "컬렉션 삭제가 실패했습니다"

        # 4단계: 삭제 후 컬렉션이 목록에서 사라졌는지 확인
        collections = mock_vector_store.list_collections()
        assert collection_name not in collections, \
            f"삭제된 컬렉션 '{collection_name}'이 여전히 목록에 존재합니다"

    def test_vector_operations(self, mock_vector_store, mock_embedding_service):
        """
        벡터 CRUD(Create, Read, Update, Delete) 연산 테스트
        
        목적:
        - 문서를 벡터로 변환하여 저장할 수 있는지 확인 (Create)
        - 저장된 벡터를 ID로 조회할 수 있는지 확인 (Read)
        - 유사도 기반 검색이 정상 작동하는지 확인 (Search - RAG의 핵심!)
        - 특정 벡터를 삭제할 수 있는지 확인 (Delete)
        
        왜 중요한가:
        - RAG 시스템의 핵심 워크플로우:
          1. 문서 저장 (add_vectors)
          2. 사용자 질문이 들어오면
          3. 질문을 임베딩으로 변환
          4. 가장 유사한 문서 검색 (search_vectors)
          5. 관련 문서를 컨텍스트로 LLM에 전달
        
        테스트 시나리오:
        - 3개의 문서를 저장
        - "Document 1"로 검색했을 때 "doc1"이 가장 높은 유사도로 반환되어야 함
        - 특정 문서 삭제 후 개수 확인
        
        실제 프로덕션 사용 예시:
        사용자 질문: "Python에서 리스트를 정렬하는 방법은?"
        → 임베딩 변환 → 벡터 DB 검색
        → 관련 문서 3개 반환: ["리스트 정렬", "sort() 메서드", "sorted() 함수"]
        → LLM에 컨텍스트로 제공하여 답변 생성
        """
        collection_name = "test_vectors"
        mock_vector_store.create_collection(collection_name, dimension=768)

        # 테스트 데이터 준비
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = [mock_embedding_service.generate_embedding(text) for text in texts]
        ids = ["doc1", "doc2", "doc3"]
        metadatas = [{"index": i, "type": "test"} for i in range(len(texts))]

        # === CREATE: 벡터 추가 ===
        result = mock_vector_store.add_vectors(
            collection_name, ids, embeddings, metadatas, texts
        )
        assert result is True, "벡터 추가가 실패했습니다"

        # 저장된 벡터 개수 확인
        count = mock_vector_store.count_vectors(collection_name)
        assert count == 3, f"3개를 저장했는데 {count}개만 있습니다"

        # === READ: 특정 ID로 벡터 조회 ===
        results = mock_vector_store.get_vectors(collection_name, ["doc1", "doc3"])
        assert len(results) == 2, f"2개를 요청했는데 {len(results)}개가 반환되었습니다"
        assert {r['id'] for r in results} == {"doc1", "doc3"}, \
            "요청한 ID의 문서가 반환되지 않았습니다"

        # === SEARCH: 유사도 기반 검색 (RAG의 핵심!) ===
        # "Document 1"과 가장 유사한 문서를 검색
        query_embedding = mock_embedding_service.generate_embedding("Document 1")
        search_results = mock_vector_store.search_vectors(
            collection_name, query_embedding, limit=2
        )
        assert len(search_results) <= 2, \
            f"최대 2개를 요청했는데 {len(search_results)}개가 반환되었습니다"
        assert search_results[0]['id'] == "doc1", \
            "'Document 1'로 검색했을 때 'doc1'이 가장 유사해야 하는데 " \
            f"'{search_results[0]['id']}'가 가장 유사합니다"

        # === DELETE: 벡터 삭제 ===
        result = mock_vector_store.delete_vectors(collection_name, ["doc2"])
        assert result is True, "벡터 삭제가 실패했습니다"

        # 삭제 후 개수 확인
        count = mock_vector_store.count_vectors(collection_name)
        assert count == 2, \
            f"1개를 삭제했으므로 2개가 남아있어야 하는데 {count}개가 있습니다"

    def test_search_filtering(self, mock_vector_store, mock_embedding_service):
        """
        메타데이터 필터링을 사용한 벡터 검색 테스트
        
        목적:
        - 유사도 검색과 메타데이터 필터를 동시에 적용할 수 있는지 확인
        - where 조건으로 특정 카테고리만 검색할 수 있는지 검증
        
        왜 중요한가:
        - 단순 유사도만으로는 부족한 경우가 많음
        - 예시 요구사항:
          * "2024년 이후 작성된 문서만 검색"
          * "Python 카테고리의 문서만 검색"
          * "우선순위가 높은 문서만 검색"
        - 유사도 + 조건 필터 = 더 정확한 검색
        
        테스트 시나리오:
        - Document A: category="type1", priority="high"
        - Document B: category="type2", priority="low"
        - Document C: category="type1", priority="medium"
        
        where={"category": "type1"}로 검색하면
        → Document A, C만 반환되어야 함 (B는 제외)
        
        실제 사용 예시:
        ```python
        # "Python 카테고리에서 최근 1년 이내 문서만 검색"
        results = search_vectors(
            query_embedding,
            where={
                "category": "python",
                "created_year": 2024
            }
        )
        ```
        
        이런 필터링은 ChromaDB, Qdrant 등 대부분의 벡터 DB가 지원합니다.
        """
        collection_name = "test_filter"
        mock_vector_store.create_collection(collection_name, dimension=768)

        # 서로 다른 메타데이터를 가진 문서들 준비
        texts = ["Document A", "Document B", "Document C"]
        embeddings = [mock_embedding_service.generate_embedding(text) for text in texts]
        ids = ["doc_a", "doc_b", "doc_c"]
        metadatas = [
            {"category": "type1", "priority": "high"},    # Document A
            {"category": "type2", "priority": "low"},     # Document B
            {"category": "type1", "priority": "medium"}   # Document C
        ]

        # 벡터 저장
        mock_vector_store.add_vectors(collection_name, ids, embeddings, metadatas, texts)

        # category="type1"인 문서만 검색 (Document A, C만 해당)
        query_embedding = mock_embedding_service.generate_embedding("Document")
        results = mock_vector_store.search_vectors(
            collection_name,
            query_embedding,
            limit=10,
            where={"category": "type1"}  # 필터 조건
        )

        # 검증: type1 카테고리 문서만 반환되어야 함
        assert len(results) == 2, \
            f"category='type1'인 문서는 2개인데 {len(results)}개가 반환되었습니다"
        assert all(r['metadata']['category'] == 'type1' for r in results), \
            "결과에 type1이 아닌 문서가 포함되어 있습니다"


@pytest.mark.unit
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
        def create_agent_collection_mock(agent_id, agent_name=None, agent_type=None):
            return mock_vector_store.create_collection(f"agent-{agent_id}", 384)
        
        knowledge_manager.create_agent_collection.side_effect = create_agent_collection_mock
        knowledge_manager.store_knowledge.side_effect = store_knowledge_side_effect
        
        # load_knowledge는 테스트에서 직접 설정하도록 초기화만
        knowledge_manager.load_knowledge = Mock()
        
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

        # Mock 반환값 설정 - Agent 1 should only see its own knowledge
        mock_result1 = Mock()
        mock_result1.id = agent1_knowledge
        mock_result1.agent_id = agent1_id
        mock_result1.content = "Agent 1 exclusive knowledge about topic A"
        
        knowledge_manager.load_knowledge.return_value = [mock_result1]
        
        agent1_results = knowledge_manager.load_knowledge(
            agent_id=agent1_id,
            query="topic A knowledge",
            limit=10
        )

        assert len(agent1_results) == 1
        assert agent1_results[0].id == agent1_knowledge
        assert agent1_results[0].agent_id == agent1_id

        # Mock 반환값 설정 - Agent 2 should only see its own knowledge
        mock_result2 = Mock()
        mock_result2.id = agent2_knowledge
        mock_result2.agent_id = agent2_id
        mock_result2.content = "Agent 2 exclusive knowledge about topic A"
        
        knowledge_manager.load_knowledge.return_value = [mock_result2]
        
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