#!/usr/bin/env python3
"""
RAG 워크플로우 간단 데모

저장 시점과 검색 시점을 명확히 보여주는 간단한 예제

실행:
    python examples/rag_workflow_simple.py
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
    print("🔄 RAG 워크플로우: 저장 vs 검색")
    print("=" * 80)
    print()
    
    store = ChromaVectorStore("./chroma-data", use_remote=False)
    embedder = SimpleEmbedding()
    collection = f"rag_simple_demo_{int(time.time())}"
    
    # ========================================================================
    # STEP 1: 저장 단계 (질문 전에 미리!)
    # ========================================================================
    print("=" * 80)
    print("STEP 1: 저장 단계 (사전 작업)")
    print("=" * 80)
    print()
    print("💾 언제? 시스템 초기 설정 시 또는 문서 생성 시")
    print("🎯 목적? 나중에 빠르게 검색하기 위해 미리 준비")
    print()
    
    # Collection 생성
    print("[1] Collection 생성...")
    store.create_collection(collection, dimension=384)
    print("✅ 완료\n")
    
    # 문서 준비
    print("[2] 회사 지식 문서 준비...")
    documents = [
        {"id": "doc1", "text": "환불은 14일 이내 가능합니다", "category": "정책"},
        {"id": "doc2", "text": "배송은 2-3일 소요됩니다", "category": "배송"},
        {"id": "doc3", "text": "로그인은 이메일로 가능합니다", "category": "FAQ"},
    ]
    
    for doc in documents:
        print(f"  📄 {doc['text']} ({doc['category']})")
    print()
    
    # 임베딩 및 저장
    print("[3] 임베딩 생성 및 ChromaDB 저장 중...")
    for doc in documents:
        embedding = embedder.encode(doc['text'])
        store.add_vectors(
            collection_name=collection,
            ids=[doc['id']],
            embeddings=[embedding],
            documents=[doc['text']],
            metadatas=[{"category": doc['category']}]
        )
        print(f"  ✅ {doc['id']} 저장")
    
    print()
    print("✅ 저장 완료! 이제 질문을 받을 준비가 되었습니다.")
    print()
    
    input("Enter를 눌러 질문 단계로...")
    
    # ========================================================================
    # STEP 2: 검색 단계 (사용자 질문 시!)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: 검색 단계 (사용자 질문 시)")
    print("=" * 80)
    print()
    print("🔍 언제? 사용자가 질문할 때마다")
    print("⚡ 속도? 매우 빠름 (0.01초)")
    print()
    
    questions = [
        "환불 방법이 궁금해요",
        "배송 기간은 얼마나 되나요?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[질문 {i}] 사용자: \"{question}\"")
        print("-" * 80)
        
        # 질문 벡터화
        print("[1] 질문을 벡터로 변환...", end=" ")
        q_embedding = embedder.encode(question)
        print("✅")
        
        # 검색
        print("[2] ChromaDB 검색...", end=" ")
        results = store.search_vectors(
            collection_name=collection,
            query_embedding=q_embedding,
            limit=1
        )
        print("✅")
        
        # 결과
        if results:
            result = results[0]
            print(f"\n[3] 검색 결과:")
            print(f"    📄 {result['content']}")
            print(f"    유사도: {result['similarity_score']:.4f}")
            print()
            print(f"[4] 🤖 AI 답변: {result['content']}")
        
        if i < len(questions):
            input("\nEnter를 눌러 다음 질문...")
    
    # ========================================================================
    # 요약
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 요약")
    print("=" * 80)
    print()
    print("저장 (Indexing):")
    print("  ⏰ 시점: 질문 전 (사전 작업)")
    print("  🐢 속도: 느림 (분~시간)")
    print("  📝 빈도: 한 번 또는 가끔")
    print()
    print("검색 (Retrieval):")
    print("  ⏰ 시점: 질문 시 (실시간)")
    print("  ⚡ 속도: 빠름 (0.01초)")
    print("  🔄 빈도: 매번")
    print()
    print("핵심:")
    print("  → 저장은 미리 해두고, 검색은 빠르게!")
    print("  → 사용자는 빠른 검색만 경험")
    print()
    
    # 정리
    response = input(f"\n테스트 컬렉션 삭제? (y/n): ")
    if response.lower() == 'y':
        store.delete_collection(collection)
        print("✅ 삭제 완료")
    
    print("\n🎉 완료!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 중단됨")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
