#!/usr/bin/env python3
"""
RAG와 LLM 연결 간단 데모

실행: python examples/rag_llm_simple.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import ChromaVectorStore
import numpy as np
import time


class SimpleEmbedding:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, text: str):
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


def main():
    print("=" * 80)
    print("🔗 RAG와 LLM 연결 데모")
    print("=" * 80)
    print()
    
    store = ChromaVectorStore("./chroma-data", use_remote=False)
    embedder = SimpleEmbedding()
    collection = f"rag_llm_demo_{int(time.time())}"
    
    # 준비: 문서 저장
    print("[준비] 문서 저장...")
    store.create_collection(collection, dimension=384)
    
    docs = [
        {"id": "1", "text": "Python sort()는 리스트를 제자리 정렬합니다"},
        {"id": "2", "text": "sorted()는 새 리스트를 반환합니다"},
    ]
    
    for doc in docs:
        embedding = embedder.encode(doc['text'])
        store.add_vectors(
            collection_name=collection,
            ids=[doc['id']],
            embeddings=[embedding],
            documents=[doc['text']],
            metadatas=[{"category": "python"}]
        )
    
    print("✅ 완료\n")
    
    # STEP 1: 질문
    print("=" * 80)
    print("STEP 1: 사용자 질문")
    print("=" * 80)
    question = "Python 리스트 정렬 방법?"
    print(f"👤 사용자: {question}\n")
    
    # STEP 2: 검색 (RAG)
    print("=" * 80)
    print("STEP 2: Vector 검색 (RAG)")
    print("=" * 80)
    q_vec = embedder.encode(question)
    results = store.search_vectors(collection, q_vec, limit=2)
    
    print("검색 결과:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['content']} (유사도: {r['similarity_score']:.4f})")
    print()
    
    # STEP 3: 컨텍스트 생성 (핵심 연결!)
    print("=" * 80)
    print("STEP 3: 컨텍스트 생성 (RAG → LLM 연결!)")
    print("=" * 80)
    
    context = "\n".join([f"[문서{i}] {r['content']}" 
                         for i, r in enumerate(results, 1)])
    
    print("생성된 컨텍스트:")
    print("-" * 80)
    print(context)
    print("-" * 80)
    print()
    
    # STEP 4: LLM 프롬프트 생성 (핵심!)
    print("=" * 80)
    print("STEP 4: LLM 프롬프트 생성")
    print("=" * 80)
    
    prompt = f"""다음 문서를 참고하여 답변하세요:

{context}

질문: {question}

답변:"""
    
    print("생성된 프롬프트:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print()
    print("💡 이 프롬프트가 LLM에 전달됩니다!")
    print()
    
    # STEP 5: LLM 응답 (시뮬레이션)
    print("=" * 80)
    print("STEP 5: LLM 답변")
    print("=" * 80)
    
    answer = """Python에서 리스트를 정렬하는 방법은 2가지입니다:

1. sort() 메서드: 원본 리스트를 제자리에서 정렬
2. sorted() 함수: 새로운 정렬된 리스트를 반환"""
    
    print("🤖 AI 답변:")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    print()
    
    # 요약
    print("=" * 80)
    print("📊 요약")
    print("=" * 80)
    print("""
흐름:
1. 질문 → 검색 (RAG가 문서 찾음)
2. 문서 → 컨텍스트 생성 ← 핵심 연결!
3. 컨텍스트 + 질문 → 프롬프트 ← 핵심 연결!
4. 프롬프트 → LLM (답변 생성)

핵심:
RAG가 찾은 문서를 LLM에게 "참고 자료"로 전달!
    """)
    
    # 정리
    response = input(f"테스트 컬렉션 삭제? (y/n): ")
    if response.lower() == 'y':
        store.delete_collection(collection)
        print("✅ 삭제 완료")
    
    print("\n🎉 완료!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ 중단")
    except Exception as e:
        print(f"❌ 에러: {e}")
