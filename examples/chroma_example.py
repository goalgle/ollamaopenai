#!/usr/bin/env python3
"""
ChromaDB VectorStore 사용 예제

이 예제는 실제 ChromaDB를 사용하여 RAG 시스템의 기본 동작을 보여줍니다.

실행 전 준비:
1. ChromaDB Docker 컨테이너 실행:
   docker run -d -v ./chroma-data:/data -p 8000:8000 chromadb/chroma

2. 필요한 패키지 설치:
   pip install chromadb numpy

3. 예제 실행:
   python examples/chroma_example.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List
from rag.vector_store import ChromaVectorStore


class SimpleEmbeddingService:
    """간단한 임베딩 서비스 (데모용)"""
    
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
    print("=" * 70)
    print("ChromaDB VectorStore 사용 예제")
    print("=" * 70)
    
    # 1. ChromaDB 연결
    print("\n[1단계] ChromaDB 로컬 모드로 연결 중...")
    try:
        vector_store = ChromaVectorStore(
            persist_directory="./chroma-data",
            use_remote=False  # 로컬 모드 사용
        )
        
        if not vector_store.health_check():
            print("❌ ChromaDB 서버에 연결할 수 없습니다!")
            print("   Docker 컨테이너를 먼저 실행하세요:")
            print("   docker run -d -v ./chroma-data:/data -p 8000:8000 chromadb/chroma")
            return
        
        print("✅ ChromaDB 연결 성공!")
        
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return
    
    # 2. 임베딩 서비스 초기화
    print("\n[2단계] 임베딩 서비스 초기화...")
    embedding_service = SimpleEmbeddingService(dimension=384)
    print("✅ 임베딩 서비스 준비 완료!")
    
    # 3. 컬렉션 생성 (에이전트의 지식 저장소)
    print("\n[3단계] 컬렉션 생성 중...")
    collection_name = "demo_python_agent"
    
    # 기존 컬렉션이 있다면 삭제
    if collection_name in vector_store.list_collections():
        print(f"   기존 컬렉션 '{collection_name}' 삭제 중...")
        vector_store.delete_collection(collection_name)
    
    vector_store.create_collection(
        collection_name=collection_name,
        dimension=384,
        metadata={"agent": "python_expert", "domain": "programming"}
    )
    print(f"✅ 컬렉션 '{collection_name}' 생성 완료!")
    
    # 4. 지식 문서 추가
    print("\n[4단계] Python 관련 지식 추가 중...")
    
    knowledge_docs = [
        {
            "id": "py_001",
            "content": "Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다.",
            "metadata": {"topic": "history", "difficulty": "beginner"}
        },
        {
            "id": "py_002",
            "content": "Python은 들여쓰기로 코드 블록을 구분하는 독특한 문법을 사용합니다.",
            "metadata": {"topic": "syntax", "difficulty": "beginner"}
        },
        {
            "id": "py_003",
            "content": "리스트 컴프리헨션은 Python의 강력한 기능으로 [x*2 for x in range(10)]처럼 사용합니다.",
            "metadata": {"topic": "advanced", "difficulty": "intermediate"}
        },
        {
            "id": "py_004",
            "content": "데코레이터는 함수를 수정하지 않고 기능을 추가하는 Python의 메타프로그래밍 기능입니다.",
            "metadata": {"topic": "advanced", "difficulty": "advanced"}
        },
        {
            "id": "py_005",
            "content": "async/await를 사용한 비동기 프로그래밍은 I/O 집약적 작업의 성능을 크게 향상시킵니다.",
            "metadata": {"topic": "async", "difficulty": "advanced"}
        }
    ]
    
    # 문서들을 임베딩하여 저장
    ids = [doc["id"] for doc in knowledge_docs]
    contents = [doc["content"] for doc in knowledge_docs]
    metadatas = [doc["metadata"] for doc in knowledge_docs]
    embeddings = [embedding_service.generate_embedding(content) for content in contents]
    
    vector_store.add_vectors(
        collection_name=collection_name,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=contents
    )
    
    print(f"✅ {len(knowledge_docs)}개의 문서 저장 완료!")
    print(f"   저장된 총 문서 수: {vector_store.count_vectors(collection_name)}개")
    
    # 5. 질문에 대한 관련 문서 검색
    print("\n[5단계] 질문에 대한 관련 문서 검색...")
    print("-" * 70)
    
    queries = [
        "Python의 역사에 대해 알려줘",
        "리스트 컴프리헨션을 어떻게 사용하나요?",
        "비동기 프로그래밍이 뭔가요?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n질문 {i}: {query}")
        print()
        
        # 질문을 임베딩으로 변환
        query_embedding = embedding_service.generate_embedding(query)
        
        # 유사한 문서 검색 (상위 2개)
        results = vector_store.search_vectors(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=2
        )
        
        print(f"  📚 찾은 관련 문서 ({len(results)}개):")
        for j, result in enumerate(results, 1):
            print(f"\n  [{j}] 문서 ID: {result['id']}")
            print(f"      유사도: {result['similarity_score']:.4f}")
            print(f"      내용: {result['content']}")
            print(f"      주제: {result['metadata']['topic']}, "
                  f"난이도: {result['metadata']['difficulty']}")
    
    # 6. 메타데이터 필터링 검색
    print("\n" + "=" * 70)
    print("[6단계] 고급 검색: 초보자용 문서만 검색")
    print("-" * 70)
    
    query = "Python 프로그래밍"
    query_embedding = embedding_service.generate_embedding(query)
    
    # 난이도가 "beginner"인 문서만 검색
    results = vector_store.search_vectors(
        collection_name=collection_name,
        query_embedding=query_embedding,
        limit=10,
        where={"difficulty": "beginner"}
    )
    
    print(f"\n  🔍 검색 조건: difficulty='beginner'")
    print(f"  📚 찾은 문서 ({len(results)}개):")
    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] {result['content']}")
        print(f"      난이도: {result['metadata']['difficulty']}")
    
    # 7. 정리
    print("\n" + "=" * 70)
    print("[7단계] 정리 중...")
    
    # 테스트 컬렉션 삭제 (선택사항)
    # vector_store.delete_collection(collection_name)
    # print(f"✅ 컬렉션 '{collection_name}' 삭제 완료!")
    
    print("\n✅ 모든 작업 완료!")
    print(f"   컬렉션 '{collection_name}'은 유지되어 다음에도 사용할 수 있습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
