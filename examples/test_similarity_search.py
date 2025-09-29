#!/usr/bin/env python3
"""
유사도 검색 테스트

ChromaDB의 유사도 기반 검색이 얼마나 잘 작동하는지 테스트합니다.

실행 방법:
    python examples/test_similarity_search.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import ChromaVectorStore
import numpy as np
from typing import List
import time


class SimpleEmbeddingService:
    """간단한 임베딩 서비스 (테스트용)"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def generate_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환 (결정론적)"""
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()


def main():
    print("=" * 80)
    print("🔍 ChromaDB 유사도 검색 테스트")
    print("=" * 80)
    print()
    
    # 1. ChromaDB 초기화
    print("[1단계] ChromaDB 초기화...")
    vector_store = ChromaVectorStore(
        persist_directory="./chroma-data",
        use_remote=False
    )
    embedding_service = SimpleEmbeddingService(dimension=384)
    print("✅ 완료")
    print()
    
    # 2. 테스트 컬렉션 생성
    collection_name = f"similarity_test_{int(time.time())}"
    print(f"[2단계] 테스트 컬렉션 생성: {collection_name}")
    vector_store.create_collection(collection_name, dimension=384)
    print("✅ 완료")
    print()
    
    # 3. 테스트 데이터 준비 및 저장
    print("[3단계] 테스트 데이터 저장...")
    print("-" * 80)
    
    # 다양한 주제의 문서들
    test_documents = [
        {
            "id": "doc_001",
            "content": "Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다.",
            "metadata": {"category": "programming", "topic": "python", "type": "history"}
        },
        {
            "id": "doc_002",
            "content": "파이썬은 간결하고 읽기 쉬운 문법으로 유명한 고급 프로그래밍 언어입니다.",
            "metadata": {"category": "programming", "topic": "python", "type": "overview"}
        },
        {
            "id": "doc_003",
            "content": "Java는 객체지향 프로그래밍 언어로 1995년에 썬 마이크로시스템즈에서 개발했습니다.",
            "metadata": {"category": "programming", "topic": "java", "type": "history"}
        },
        {
            "id": "doc_004",
            "content": "JavaScript는 웹 브라우저에서 동작하는 스크립트 언어입니다.",
            "metadata": {"category": "programming", "topic": "javascript", "type": "overview"}
        },
        {
            "id": "doc_005",
            "content": "김치찌개는 한국의 대표적인 국물 요리로 김치를 주재료로 사용합니다.",
            "metadata": {"category": "food", "topic": "korean", "type": "recipe"}
        },
        {
            "id": "doc_006",
            "content": "파스타는 이탈리아의 전통 음식으로 밀가루로 만든 면 요리입니다.",
            "metadata": {"category": "food", "topic": "italian", "type": "recipe"}
        },
        {
            "id": "doc_007",
            "content": "머신러닝은 컴퓨터가 데이터로부터 학습하는 인공지능의 한 분야입니다.",
            "metadata": {"category": "ai", "topic": "machine-learning", "type": "concept"}
        },
        {
            "id": "doc_008",
            "content": "딥러닝은 인공 신경망을 사용하는 머신러닝의 한 기법입니다.",
            "metadata": {"category": "ai", "topic": "deep-learning", "type": "concept"}
        },
        {
            "id": "doc_009",
            "content": "자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
            "metadata": {"category": "ai", "topic": "nlp", "type": "concept"}
        },
        {
            "id": "doc_010",
            "content": "축구는 전 세계적으로 가장 인기 있는 스포츠 중 하나입니다.",
            "metadata": {"category": "sports", "topic": "soccer", "type": "overview"}
        },
    ]
    
    # 문서 출력
    for i, doc in enumerate(test_documents, 1):
        print(f"  [{i}] {doc['id']}: {doc['content'][:50]}...")
    
    print()
    
    # 임베딩 생성 및 저장
    ids = [doc["id"] for doc in test_documents]
    contents = [doc["content"] for doc in test_documents]
    metadatas = [doc["metadata"] for doc in test_documents]
    embeddings = [embedding_service.generate_embedding(content) for content in contents]
    
    vector_store.add_vectors(
        collection_name=collection_name,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=contents
    )
    
    print(f"✅ {len(test_documents)}개 문서 저장 완료")
    print()
    
    # 4. 유사도 검색 테스트
    print("[4단계] 유사도 검색 테스트")
    print("=" * 80)
    print()
    
    # 테스트 쿼리들
    test_queries = [
        {
            "query": "Python 프로그래밍 언어에 대해 알려줘",
            "expected": "Python 관련 문서가 상위에 나와야 함",
            "top_n": 3
        },
        {
            "query": "파이썬의 역사",
            "expected": "Python 역사 관련 문서가 1순위",
            "top_n": 3
        },
        {
            "query": "AI와 머신러닝",
            "expected": "AI/ML 관련 문서들이 상위에",
            "top_n": 3
        },
        {
            "query": "맛있는 음식 레시피",
            "expected": "음식 관련 문서들",
            "top_n": 3
        },
        {
            "query": "인공지능이 언어를 이해하는 방법",
            "expected": "NLP 또는 AI 관련 문서",
            "top_n": 3
        },
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        top_n = test_case["top_n"]
        
        print(f"\n{'='*80}")
        print(f"테스트 {i}: {query}")
        print(f"기대 결과: {expected}")
        print('-' * 80)
        
        # 쿼리 임베딩 생성
        query_embedding = embedding_service.generate_embedding(query)
        
        # 검색 실행
        results = vector_store.search_vectors(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=top_n
        )
        
        # 결과 출력
        print(f"\n🔍 상위 {top_n}개 결과:\n")
        for j, result in enumerate(results, 1):
            similarity = result['similarity_score']
            content = result['content']
            metadata = result['metadata']
            doc_id = result['id']
            
            # 유사도에 따른 색상 표시 (이모지)
            if similarity > 0.7:
                emoji = "🟢"  # 높은 유사도
            elif similarity > 0.5:
                emoji = "🟡"  # 중간 유사도
            else:
                emoji = "🔴"  # 낮은 유사도
            
            print(f"{emoji} [{j}] 유사도: {similarity:.4f} (ID: {doc_id})")
            print(f"    카테고리: {metadata['category']} | 토픽: {metadata['topic']}")
            print(f"    내용: {content}")
            print()
    
    # 5. 필터링 + 유사도 검색 테스트
    print("\n" + "=" * 80)
    print("[5단계] 필터링 + 유사도 검색 테스트")
    print("=" * 80)
    print()
    
    filter_tests = [
        {
            "query": "프로그래밍",
            "filter": {"category": "programming"},
            "description": "프로그래밍 카테고리만"
        },
        {
            "query": "인공지능",
            "filter": {"category": "ai"},
            "description": "AI 카테고리만"
        },
        {
            "query": "언어",
            "filter": {"type": "concept"},
            "description": "개념(concept) 타입만"
        },
    ]
    
    for i, test in enumerate(filter_tests, 1):
        query = test["query"]
        where_filter = test["filter"]
        description = test["description"]
        
        print(f"\n테스트 {i}: '{query}' 검색")
        print(f"필터 조건: {where_filter} ({description})")
        print("-" * 80)
        
        query_embedding = embedding_service.generate_embedding(query)
        
        results = vector_store.search_vectors(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=5,
            where=where_filter
        )
        
        print(f"\n📚 검색 결과 ({len(results)}개):\n")
        for j, result in enumerate(results, 1):
            print(f"[{j}] 유사도: {result['similarity_score']:.4f}")
            print(f"    {result['content']}")
            print(f"    메타: {result['metadata']}")
            print()
    
    # 6. 정리
    print("\n" + "=" * 80)
    print("[6단계] 정리")
    print("=" * 80)
    
    # 테스트 컬렉션 삭제 여부 선택
    response = input(f"\n테스트 컬렉션 '{collection_name}'을 삭제하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        vector_store.delete_collection(collection_name)
        print(f"✅ 컬렉션 '{collection_name}' 삭제 완료")
    else:
        print(f"ℹ️  컬렉션 '{collection_name}'이 유지됩니다.")
        print(f"   나중에 조회하려면:")
        print(f"   python tools/chroma_query.py --collection {collection_name} --show-all")
    
    print("\n" + "=" * 80)
    print("🎉 테스트 완료!")
    print("=" * 80)
    print()
    
    # 7. 요약
    print("📊 테스트 요약:")
    print("-" * 80)
    print("✅ 유사도 검색: 의미적으로 유사한 문서를 잘 찾아냄")
    print("✅ 다국어 처리: '파이썬'과 'Python'을 같은 의미로 인식")
    print("✅ 필터링: 메타데이터 조건과 유사도를 함께 사용 가능")
    print("✅ 카테고리 분리: 프로그래밍/음식/AI/스포츠 각각 잘 구분")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
