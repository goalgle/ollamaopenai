#!/usr/bin/env python3
"""
벡터 유사도 검색 원리 설명

텍스트가 어떻게 벡터로 변환되고, 유사도가 계산되는지
시각적으로 보여주는 예제입니다.

실행:
    python examples/vector_similarity_explained.py
"""

import numpy as np
from typing import List, Tuple
import math


def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def simple_text_to_vector(text: str) -> List[float]:
    """
    간단한 방법으로 텍스트를 벡터로 변환 (설명용)
    
    실제로는 BERT, OpenAI 같은 복잡한 모델을 사용하지만,
    여기서는 이해를 돕기 위해 간단한 방법 사용
    """
    # 단어 빈도를 기반으로 벡터 생성
    keywords = ['python', 'java', 'programming', 'food', 'cooking', 'music', 'sports']
    
    text_lower = text.lower()
    vector = []
    
    for keyword in keywords:
        # 키워드가 텍스트에 포함되어 있으면 1, 아니면 0
        vector.append(1.0 if keyword in text_lower else 0.0)
    
    return vector


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    코사인 유사도 계산
    
    두 벡터 사이의 각도를 측정 (0도 = 완전 같음, 90도 = 완전 다름)
    """
    # 내적 (dot product)
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # 벡터의 크기 (magnitude)
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # 코사인 유사도
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    유클리드 거리 계산
    
    두 벡터 사이의 직선 거리 (작을수록 유사)
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def visualize_vector(text: str, vector: List[float], keywords: List[str]):
    """벡터를 시각적으로 표현"""
    print(f"📝 텍스트: \"{text}\"")
    print(f"🔢 벡터: {vector}")
    print("\n시각화:")
    
    for i, (keyword, value) in enumerate(zip(keywords, vector)):
        bar = "█" * int(value * 10)
        print(f"  {keyword:12} : {bar} ({value:.1f})")
    print()


def main():
    print("=" * 80)
    print("🧮 벡터 유사도 검색의 원리")
    print("=" * 80)
    
    # ============================================================================
    # STEP 1: 텍스트를 벡터로 변환
    # ============================================================================
    print_section("STEP 1: 텍스트를 벡터(숫자 배열)로 변환")
    
    print("텍스트는 컴퓨터가 이해할 수 없습니다.")
    print("→ 숫자로 변환해야 계산 가능!")
    print()
    
    keywords = ['python', 'java', 'programming', 'food', 'cooking', 'music', 'sports']
    print(f"기준 키워드: {keywords}")
    print("(실제로는 수천~수만 개의 차원을 사용)")
    print()
    
    # 예제 문서들
    doc1 = "Python is a great programming language"
    doc2 = "I love programming in Python"
    doc3 = "Java is also a programming language"
    doc4 = "Italian food and cooking recipes"
    
    vec1 = simple_text_to_vector(doc1)
    vec2 = simple_text_to_vector(doc2)
    vec3 = simple_text_to_vector(doc3)
    vec4 = simple_text_to_vector(doc4)
    
    visualize_vector(doc1, vec1, keywords)
    visualize_vector(doc2, vec2, keywords)
    visualize_vector(doc3, vec3, keywords)
    visualize_vector(doc4, vec4, keywords)
    
    # ============================================================================
    # STEP 2: 유사도 계산
    # ============================================================================
    print_section("STEP 2: 벡터 간 유사도 계산")
    
    print("💡 핵심 아이디어:")
    print("   - 비슷한 의미의 텍스트 → 비슷한 벡터")
    print("   - 다른 의미의 텍스트 → 다른 벡터")
    print()
    
    print("📐 코사인 유사도 (Cosine Similarity)")
    print("   - 두 벡터 사이의 각도를 측정")
    print("   - 1.0 = 완전히 같은 방향 (매우 유사)")
    print("   - 0.0 = 직각 (무관)")
    print("   - -1.0 = 반대 방향 (반대)")
    print()
    
    # 유사도 계산
    sim_1_2 = cosine_similarity(vec1, vec2)
    sim_1_3 = cosine_similarity(vec1, vec3)
    sim_1_4 = cosine_similarity(vec1, vec4)
    
    print("유사도 계산 결과:")
    print("-" * 80)
    print(f"Doc1 vs Doc2: {sim_1_2:.4f}  ← Python + programming (매우 유사!)")
    print(f"  \"{doc1}\"")
    print(f"  \"{doc2}\"")
    print()
    
    print(f"Doc1 vs Doc3: {sim_1_3:.4f}  ← programming 공통 (조금 유사)")
    print(f"  \"{doc1}\"")
    print(f"  \"{doc3}\"")
    print()
    
    print(f"Doc1 vs Doc4: {sim_1_4:.4f}  ← 완전히 다른 주제 (유사하지 않음)")
    print(f"  \"{doc1}\"")
    print(f"  \"{doc4}\"")
    print()
    
    # ============================================================================
    # STEP 3: 실제 검색 시뮬레이션
    # ============================================================================
    print_section("STEP 3: 검색 시뮬레이션")
    
    # 문서 DB
    documents = [
        {"id": "doc1", "text": doc1, "vector": vec1},
        {"id": "doc2", "text": doc2, "vector": vec2},
        {"id": "doc3", "text": doc3, "vector": vec3},
        {"id": "doc4", "text": doc4, "vector": vec4},
    ]
    
    # 검색 쿼리
    query = "Python programming tutorial"
    print(f"🔍 검색어: \"{query}\"")
    print()
    
    # 쿼리를 벡터로 변환
    query_vector = simple_text_to_vector(query)
    visualize_vector(query, query_vector, keywords)
    
    # 모든 문서와 유사도 계산
    print("📊 모든 문서와의 유사도:")
    print("-" * 80)
    
    similarities = []
    for doc in documents:
        sim = cosine_similarity(query_vector, doc["vector"])
        similarities.append((doc, sim))
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\n검색 결과 (유사도 높은 순):\n")
    for i, (doc, sim) in enumerate(similarities, 1):
        if sim > 0.7:
            emoji = "🟢"  # 매우 관련
        elif sim > 0.3:
            emoji = "🟡"  # 조금 관련
        else:
            emoji = "🔴"  # 무관
        
        print(f"{emoji} [{i}] 유사도: {sim:.4f}")
        print(f"    {doc['id']}: \"{doc['text']}\"")
        print()
    
    # ============================================================================
    # STEP 4: 고차원 벡터의 위력
    # ============================================================================
    print_section("STEP 4: 실제 임베딩의 위력")
    
    print("🎯 우리가 사용한 간단한 방법 (7차원):")
    print("   - 키워드 7개로만 표현")
    print("   - 제한적인 의미 파악")
    print()
    
    print("⚡ 실제 임베딩 모델 (384~1536차원):")
    print("   - OpenAI: 1536차원")
    print("   - BERT: 768차원")
    print("   - Sentence Transformers: 384차원")
    print()
    
    print("💡 고차원의 장점:")
    print("   ✅ 미묘한 의미 차이 포착")
    print("   ✅ 동의어 자동 인식")
    print("   ✅ 문맥 이해")
    print("   ✅ 다국어 지원")
    print()
    
    print("예시:")
    print("   '투자' ≈ '재테크' ≈ '자산관리' ≈ 'investment'")
    print("   → 모두 비슷한 벡터로 변환!")
    print()
    
    # ============================================================================
    # STEP 5: 실제 ChromaDB의 동작
    # ============================================================================
    print_section("STEP 5: ChromaDB의 실제 동작")
    
    print("📚 ChromaDB가 하는 일:")
    print()
    print("1️⃣  문서 저장 시:")
    print("   문서 → 임베딩 모델 → 벡터(384차원) → DB 저장")
    print()
    print("2️⃣  검색 시:")
    print("   질문 → 임베딩 모델 → 쿼리 벡터(384차원)")
    print("   → 모든 문서 벡터와 유사도 계산")
    print("   → 유사도 높은 순으로 반환")
    print()
    print("3️⃣  최적화:")
    print("   - 수백만 개 문서 중에서도 빠르게 검색")
    print("   - 근사 최근접 이웃(ANN) 알고리즘 사용")
    print("   - HNSW, IVF 같은 인덱싱 기법")
    print()
    
    # ============================================================================
    # 요약
    # ============================================================================
    print_section("📌 핵심 요약")
    
    print("✅ 텍스트 → 벡터(숫자 배열) 변환")
    print("   └─ 임베딩 모델 사용 (BERT, OpenAI 등)")
    print()
    print("✅ 비슷한 의미 = 비슷한 벡터")
    print("   └─ '투자', '재테크', 'investment' 모두 비슷한 위치")
    print()
    print("✅ 코사인 유사도로 벡터 간 거리 측정")
    print("   └─ 1.0에 가까울수록 유사")
    print()
    print("✅ 키워드 없이도 의미로 검색 가능!")
    print("   └─ 전통 검색의 한계 극복")
    print()
    
    print("-" * 80)
    print("💡 다음에 해볼 것:")
    print("   python examples/test_similarity_search.py  ← 실제 검색 테스트")
    print("=" * 80)


if __name__ == "__main__":
    main()
