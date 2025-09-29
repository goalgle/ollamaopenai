#!/usr/bin/env python3
"""
ChromaDB 통합 테스트
실제 ChromaDB를 사용하여 VectorStore 기능 검증
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from typing import List
from rag.vector_store import ChromaVectorStore


class TestEmbeddingService:
    """간단한 임베딩 서비스 (테스트용)"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def generate_embedding(self, text: str) -> List[float]:
        """결정론적 임베딩 생성"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # 텍스트 해시 기반 시드
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)
        
        # 정규화
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()


@pytest.fixture
def chroma_store():
    """
    ChromaDB VectorStore 픽스처
    
    로컬 모드 사용 (Docker 불필요)
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ChromaVectorStore(
            persist_directory=tmp_dir,
            use_remote=False  # 로컬 모드
        )
        
        yield store


@pytest.fixture
def embedding_service():
    """임베딩 서비스 픽스처"""
    return TestEmbeddingService(dimension=384)


@pytest.mark.integration
@pytest.mark.chroma
class TestChromaVectorStore:
    """ChromaDB VectorStore 통합 테스트"""

    def test_health_check(self, chroma_store):
        """ChromaDB 서버 연결 상태 확인"""
        assert chroma_store.health_check() is True

    def test_collection_management(self, chroma_store):
        """컬렉션 생성 및 삭제 기능 테스트"""
        import time
        collection_name = f"test_collection_{int(time.time())}"
        
        try:
            result = chroma_store.create_collection(collection_name, dimension=384)
            assert result is True
            
            collections = chroma_store.list_collections()
            assert collection_name in collections
            
            result = chroma_store.delete_collection(collection_name)
            assert result is True
            
            collections = chroma_store.list_collections()
            assert collection_name not in collections
                
        finally:
            try:
                chroma_store.delete_collection(collection_name)
            except:
                pass

    def test_vector_operations(self, chroma_store, embedding_service):
        """벡터 CRUD 연산 테스트"""
        import time
        collection_name = f"test_vectors_{int(time.time())}"
        
        try:
            chroma_store.create_collection(collection_name, dimension=384)
            
            texts = ["Document 1", "Document 2", "Document 3"]
            embeddings = [embedding_service.generate_embedding(text) for text in texts]
            ids = ["doc1", "doc2", "doc3"]
            metadatas = [{"index": i, "type": "test"} for i in range(len(texts))]
            
            result = chroma_store.add_vectors(collection_name, ids, embeddings, metadatas, texts)
            assert result is True
            
            count = chroma_store.count_vectors(collection_name)
            assert count == 3
            
            results = chroma_store.get_vectors(collection_name, ["doc1", "doc3"])
            assert len(results) == 2
            
            query_embedding = embedding_service.generate_embedding("Document 1")
            search_results = chroma_store.search_vectors(collection_name, query_embedding, limit=2)
            assert len(search_results) > 0
            assert search_results[0]['id'] == "doc1"
            
            print(f"\n검색 결과:")
            for i, result in enumerate(search_results):
                print(f"  {i+1}. {result['id']}: {result['similarity_score']:.4f}")
            
        finally:
            try:
                chroma_store.delete_collection(collection_name)
            except:
                pass

    def test_search_filtering(self, chroma_store, embedding_service):
        """메타데이터 필터링 검색 테스트"""
        import time
        collection_name = f"test_filter_{int(time.time())}"
        
        try:
            chroma_store.create_collection(collection_name, dimension=384)
            
            texts = ["Document A", "Document B", "Document C"]
            embeddings = [embedding_service.generate_embedding(text) for text in texts]
            ids = ["doc_a", "doc_b", "doc_c"]
            metadatas = [
                {"category": "type1", "priority": "high"},
                {"category": "type2", "priority": "low"},
                {"category": "type1", "priority": "medium"}
            ]
            
            chroma_store.add_vectors(collection_name, ids, embeddings, metadatas, texts)
            
            query_embedding = embedding_service.generate_embedding("Document")
            results = chroma_store.search_vectors(
                collection_name,
                query_embedding,
                limit=10,
                where={"category": "type1"}
            )
            
            assert len(results) == 2
            assert all(r['metadata']['category'] == 'type1' for r in results)
            
        finally:
            try:
                chroma_store.delete_collection(collection_name)
            except:
                pass


def test_chroma_connection():
    """ChromaDB 연결 테스트 (로컬 모드)"""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ChromaVectorStore(
                persist_directory=tmp_dir,
                use_remote=False
            )
            
            assert store.health_check()
            
            print("\n✅ ChromaDB 연결 성공!")
            print("   모드: 로컬 (파일 시스템)")
            
            collections = store.list_collections()
            print(f"   기존 컬렉션 수: {len(collections)}개")
        
    except Exception as e:
        print("\n❌ ChromaDB 연결 실패!")
        print(f"   에러: {e}")
        raise


if __name__ == "__main__":
    test_chroma_connection()
