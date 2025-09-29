#!/usr/bin/env python3
"""
ChromaDB 개념 실습 예제

Collection, Document, Metadata의 개념을
RDB와 비교하며 이해하는 예제입니다.

실행:
    python examples/chromadb_concepts_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import ChromaVectorStore
import numpy as np
from typing import List
import time


class SimpleEmbedding:
    """간단한 임베딩 (테스트용)"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, text: str) -> List[float]:
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print("=" * 80)
    print("🎨 ChromaDB 개념 실습")
    print("=" * 80)
    print()
    
    # 초기화
    print("[초기화]")
    store = ChromaVectorStore(
        persist_directory="./chroma-data",
        use_remote=False
    )
    embedder = SimpleEmbedding()
    print("✅ ChromaDB 준비 완료\n")
    
    # PART 1: Collection = RDB의 Table
    print_header("PART 1: Collection 개념 (RDB의 Table)")
    
    print("💡 비유:")
    print("   RDB Database")
    print("     └─ Table: users, products, orders")
    print()
    print("   ChromaDB Instance")
    print("     └─ Collection: user_profiles, product_docs, reviews")
    print()
    
    print("📚 3개의 Collection 생성 (= 3개의 Table)\n")
    
    collections = [
        ("tech_docs", "기술 문서"),
        ("customer_reviews", "고객 리뷰"),
        ("company_policies", "회사 정책")
    ]
    
    for name, description in collections:
        full_name = f"demo_{name}_{int(time.time())}"
        store.create_collection(full_name, dimension=384)
        print(f"  ✅ Collection '{name}' 생성 → {description}")
    
    print()
    input("Enter를 눌러 계속...")
    
    # PART 2: Document = RDB의 Row
    print_header("PART 2: Document 개념 (RDB의 Row)")
    
    print("💡 비교:\n")
    
    print("📊 [RDB - products 테이블]")
    print("-" * 80)
    print("| ID  | name   | description              | price | category    |")
    print("|-----|--------|--------------------------|-------|-------------|")
    print("| 1   | Laptop | High-performance laptop  | 1500  | electronics |")
    print("| 2   | Mouse  | Wireless mouse           | 30    | accessories |")
    print("-" * 80)
    print()
    
    print("🎨 [ChromaDB - product_docs 컬렉션]")
    print("-" * 80)
    
    collection_name = f"demo_products_{int(time.time())}"
    store.create_collection(collection_name, dimension=384)
    
    products = [
        {
            "id": "prod_001",
            "text": "High-performance laptop for programming and gaming",
            "metadata": {
                "name": "Laptop",
                "price": 1500,
                "category": "electronics"
            }
        },
        {
            "id": "prod_002",
            "text": "Wireless mouse with ergonomic design",
            "metadata": {
                "name": "Mouse",
                "price": 30,
                "category": "accessories"
            }
        }
    ]
    
    for product in products:
        embedding = embedder.encode(product['text'])
        
        store.add_vectors(
            collection_name=collection_name,
            ids=[product['id']],
            embeddings=[embedding],
            documents=[product['text']],
            metadatas=[product['metadata']]
        )
        
        print(f"Document:")
        print(f"  ├─ id: {product['id']}")
        print(f"  ├─ embedding: [0.1, 0.5, ...] (384차원 벡터)")
        print(f"  ├─ document: \"{product['text']}\"")
        print(f"  └─ metadata: {product['metadata']}")
        print()
    
    print("-" * 80)
    print()
    print("💡 핵심 차이:")
    print("   RDB: 정형 데이터 (숫자, 텍스트)")
    print("   ChromaDB: 벡터 + 메타데이터 (의미 검색 가능!)")
    print()
    
    input("Enter를 눌러 계속...")
    
    # PART 3: 검색 비교
    print_header("PART 3: 검색 비교 (SQL vs Vector Search)")
    
    print("🔍 시나리오: '노트북' 검색\n")
    
    print("📊 [RDB - SQL 검색]")
    print("-" * 80)
    print("SELECT * FROM products WHERE description LIKE '%laptop%';")
    print()
    print("결과:")
    print("  ✅ Laptop (정확히 'laptop' 포함)")
    print("  ❌ Mouse (단어 없음)")
    print()
    print("한계:")
    print("  ❌ '노트북'으로 검색 → 못 찾음")
    print("  ❌ '컴퓨터'로 검색 → 못 찾음")
    print("-" * 80)
    print()
    
    print("🎨 [ChromaDB - Vector 검색]")
    print("-" * 80)
    
    for query in ["노트북", "컴퓨터", "gaming laptop"]:
        print(f"검색어: \"{query}\"")
        query_embedding = embedder.encode(query)
        
        results = store.search_vectors(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=2
        )
        
        print(f"결과:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['name']} "
                  f"(유사도: {result['similarity_score']:.4f})")
        print()
    
    print("장점:")
    print("  ✅ 언어 무관 (한글/영어 모두)")
    print("  ✅ 의미 기반 검색")
    print("-" * 80)
    
    print()
    print("=" * 80)
    print("🎉 ChromaDB 개념 학습 완료!")
    print("=" * 80)
    print()
    print("핵심 요약:")
    print("  1. Collection = Table (데이터 그룹)")
    print("  2. Document = Row (개별 항목)")
    print("  3. Embedding = Vector (의미를 숫자로)")
    print("  4. Metadata = 추가 속성 (필터링용)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 에러: {e}")
        import traceback
        traceback.print_exc()
