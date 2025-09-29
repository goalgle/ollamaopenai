#!/usr/bin/env python3
"""
유사도 검색 기반 문서 요약 예제

대용량 문서에서 특정 주제에 대한 정보만 추출하여 요약하는 예제입니다.

사용 사례:
- 100페이지 보고서에서 "재무 실적" 관련 내용만 추출
- 긴 논문에서 "실험 방법" 부분만 찾기
- 법률 문서에서 특정 조항 찾기
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import ChromaVectorStore
import numpy as np
from typing import List
import time


class SimpleEmbeddingService:
    """간단한 임베딩 서비스"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def generate_embedding(self, text: str) -> List[float]:
        text_hash = hash(text) % (2**31)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.dimension)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


def chunk_document(text: str, chunk_size: int = 200) -> List[str]:
    """문서를 작은 청크로 분할"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def main():
    print("=" * 80)
    print("📝 유사도 검색 기반 문서 요약 시스템")
    print("=" * 80)
    print()
    
    # 1. 초기화
    print("[1단계] 시스템 초기화...")
    vector_store = ChromaVectorStore(
        persist_directory="./chroma-data",
        use_remote=False
    )
    embedding_service = SimpleEmbeddingService()
    collection_name = f"doc_summary_test_{int(time.time())}"
    vector_store.create_collection(collection_name, dimension=384)
    print("✅ 완료\n")
    
    # 2. 가상의 대용량 보고서 (실제로는 수백 페이지)
    print("[2단계] 대용량 문서 준비...")
    print("-" * 80)
    
    # 가상의 기업 분기 보고서
    long_document = """
    2024년 3분기 실적 보고서
    
    [경영 실적 요약]
    당사는 2024년 3분기에 매출 1,250억원을 달성하였으며, 이는 전년 동기 대비 15% 증가한 수치입니다.
    영업이익은 180억원으로 전년 대비 20% 증가하였으며, 순이익은 140억원을 기록했습니다.
    주요 성장 동력은 신제품 출시와 해외 시장 확대에 있었습니다.
    
    [제품 개발 현황]
    연구개발팀은 AI 기반 추천 시스템 개발을 완료하였으며, 베타 테스트 중입니다.
    새로운 모바일 앱은 10만 다운로드를 돌파했으며 사용자 만족도는 4.5점입니다.
    차세대 플랫폼 개발에 50억원을 투자하였으며, 2025년 1분기 출시 예정입니다.
    향후 머신러닝과 자연어처리 기술을 활용한 신규 서비스를 준비 중입니다.
    
    [시장 분석]
    국내 시장 점유율은 23%로 업계 2위를 유지하고 있습니다.
    경쟁사 대비 기술력에서 우위를 점하고 있으나, 마케팅 예산은 부족한 상황입니다.
    글로벌 시장 진출을 위해 동남아시아 3개국에 지사를 설립할 예정입니다.
    시장 조사 결과, 고객들은 가격보다 품질과 편의성을 중요하게 생각하는 것으로 나타났습니다.
    
    [재무 상태]
    총 자산은 5,200억원이며 부채비율은 45%로 안정적인 수준입니다.
    현금 및 현금성 자산은 1,100억원으로 충분한 유동성을 확보하고 있습니다.
    신규 투자를 위해 300억원 규모의 회사채 발행을 검토 중입니다.
    배당금은 주당 500원으로 작년 대비 100원 증가하였습니다.
    
    [인사 조직]
    3분기 신규 채용 인원은 120명이며, 총 직원 수는 1,850명입니다.
    평균 근속연수는 5.2년이며, 이직률은 8%로 업계 평균보다 낮습니다.
    직원 교육에 20억원을 투자하였으며, 리더십 프로그램을 강화했습니다.
    재택근무와 유연근무제를 확대하여 직원 만족도가 향상되었습니다.
    
    [리스크 관리]
    환율 변동에 대비하여 헷징 전략을 수립하였습니다.
    사이버 보안 시스템을 강화하고 개인정보 보호 조치를 강화했습니다.
    공급망 다변화를 통해 원자재 수급 리스크를 최소화하고 있습니다.
    규제 변화에 대응하기 위해 법무팀을 확대 개편하였습니다.
    
    [향후 전망]
    4분기에는 연말 시즌 효과로 매출 1,400억원을 목표로 하고 있습니다.
    2025년에는 신사업 진출과 M&A를 통해 성장을 가속화할 계획입니다.
    디지털 전환과 AI 기술 도입으로 운영 효율성을 20% 개선할 예정입니다.
    지속가능경영을 강화하고 ESG 평가 등급을 상향 조정하는 것이 목표입니다.
    """
    
    print(f"문서 길이: {len(long_document)}자")
    print(f"문서 미리보기:\n{long_document[:200]}...\n")
    
    # 3. 문서를 작은 청크로 분할
    print("[3단계] 문서를 청크로 분할...")
    chunks = chunk_document(long_document, chunk_size=50)
    print(f"✅ {len(chunks)}개 청크로 분할 완료\n")
    
    # 4. 청크를 벡터 DB에 저장
    print("[4단계] 벡터 DB에 저장...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [embedding_service.generate_embedding(chunk) for chunk in chunks]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    
    vector_store.add_vectors(
        collection_name=collection_name,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks
    )
    print(f"✅ {len(chunks)}개 청크 저장 완료\n")
    
    # 5. 특정 주제에 대한 정보만 추출
    print("=" * 80)
    print("[5단계] 주제별 정보 추출 테스트")
    print("=" * 80)
    print()
    
    queries = [
        {
            "topic": "재무 실적",
            "query": "매출과 영업이익 재무 성과",
            "description": "재무 관련 수치와 실적"
        },
        {
            "topic": "제품 개발",
            "query": "신제품 개발 연구 기술 혁신",
            "description": "R&D와 제품 개발 현황"
        },
        {
            "topic": "인사 조직",
            "query": "직원 채용 인사 조직 문화",
            "description": "인력 관리 및 조직 문화"
        },
        {
            "topic": "향후 계획",
            "query": "미래 전망 계획 목표 전략",
            "description": "향후 사업 계획 및 전망"
        },
    ]
    
    for i, test in enumerate(queries, 1):
        topic = test["topic"]
        query = test["query"]
        description = test["description"]
        
        print(f"\n{'='*80}")
        print(f"🔍 주제 {i}: {topic}")
        print(f"   설명: {description}")
        print(f"   검색어: {query}")
        print('-' * 80)
        
        # 유사도 검색
        query_embedding = embedding_service.generate_embedding(query)
        results = vector_store.search_vectors(
            collection_name=collection_name,
            query_embedding=query_embedding,
            limit=3  # 상위 3개 관련 청크만
        )
        
        print(f"\n📄 관련 내용 ({len(results)}개 청크):\n")
        
        # 요약 생성 (관련 청크들을 합침)
        summary_parts = []
        for j, result in enumerate(results, 1):
            similarity = result['similarity_score']
            content = result['content'].strip()
            
            print(f"[청크 {j}] 유사도: {similarity:.4f}")
            print(f"{content}\n")
            
            if similarity > 0.5:  # 유사도가 높은 것만 요약에 포함
                summary_parts.append(content)
        
        # 요약본
        if summary_parts:
            summary = " ".join(summary_parts)
            print(f"💡 {topic} 요약:")
            print(f"   {summary[:300]}...")
    
    # 6. 정리
    print("\n" + "=" * 80)
    print("[6단계] 정리")
    print("=" * 80)
    
    response = input(f"\n테스트 컬렉션을 삭제하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        vector_store.delete_collection(collection_name)
        print(f"✅ 컬렉션 삭제 완료")
    else:
        print(f"ℹ️  컬렉션 유지: {collection_name}")
    
    print("\n" + "=" * 80)
    print("🎉 테스트 완료!")
    print("=" * 80)
    print()
    
    print("📊 활용 사례:")
    print("-" * 80)
    print("✅ 대용량 보고서에서 특정 주제만 추출")
    print("✅ 긴 문서를 주제별로 자동 분류")
    print("✅ 관련 정보만 모아서 요약 생성")
    print("✅ 키워드 없이도 의미적으로 관련된 내용 찾기")
    print()
    print("💡 실제 활용:")
    print("   - 법률 문서에서 특정 조항 찾기")
    print("   - 논문에서 실험 방법론만 추출")
    print("   - 고객 리뷰에서 특정 기능 관련 피드백 수집")
    print("   - 뉴스 아카이브에서 특정 사건 관련 기사 찾기")
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
