#!/usr/bin/env python3
"""
주식 트레이딩 에이전트 예제

실제 주식 에이전트가 RAG를 어떻게 활용하는지 보여주는 데모

실행:
    python examples/stock_agent_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import ChromaVectorStore
from rag.chroma_util import ChromaUtil
import numpy as np
from typing import List, Dict
import time
from datetime import datetime


class SimpleEmbedding:
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


class StockAgent:
    """주식 트레이딩 AI 에이전트"""
    
    def __init__(self, rag_dir: str = "./stock-rag-data"):
        self.store = ChromaVectorStore(rag_dir, use_remote=False)
        self.chroma_util = ChromaUtil(rag_dir, use_remote=False)
        self.embedder = SimpleEmbedding()
        self.timestamp = int(time.time())
    
    def setup_knowledge_base(self):
        """지식 베이스 초기 구축"""
        
        print("=" * 80)
        print("📚 주식 에이전트 지식 베이스 구축")
        print("=" * 80)
        print()
        
        # ================================================================
        # Collection 1: 투자 지식
        # ================================================================
        print("[1/4] 투자 지식 베이스 구축 중...")
        
        knowledge_collection = f"trading_knowledge_{self.timestamp}"
        self.store.create_collection(knowledge_collection, dimension=384)
        
        trading_knowledge = [
            {
                "id": "rsi_indicator",
                "title": "RSI (상대강도지수)",
                "content": """
RSI는 0~100 범위의 모멘텀 지표입니다.
- 70 이상: 과매수 구간 (매도 고려)
- 30 이하: 과매도 구간 (매수 고려)
- 50: 중립
일반적으로 14일 기준을 사용합니다.
                """,
                "category": "기술적지표"
            },
            {
                "id": "macd_indicator",
                "title": "MACD",
                "content": """
MACD는 추세를 파악하는 지표입니다.
- 골든크로스: MACD선이 시그널선을 상향 돌파 (매수 신호)
- 데드크로스: MACD선이 시그널선을 하향 돌파 (매도 신호)
                """,
                "category": "기술적지표"
            },
            {
                "id": "value_investing",
                "title": "가치투자 전략",
                "content": """
기업의 내재가치 대비 저평가된 주식을 매수하는 전략입니다.
주요 지표: PER(주가수익비율), PBR(주가순자산비율), ROE(자기자본이익률)
- PER 낮을수록 저평가
- PBR 1 미만이면 청산가치보다 낮음
- ROE 높을수록 수익성 좋음
                """,
                "category": "투자전략"
            },
            {
                "id": "risk_management",
                "title": "리스크 관리",
                "content": """
성공적인 투자를 위한 리스크 관리 원칙:
1. 손절매 철저히 (-5~10% 수준)
2. 분산투자 (계란을 한 바구니에 담지 말 것)
3. 포지션 크기 조절 (한 종목 10% 이하)
4. 감정적 거래 금지
                """,
                "category": "리스크관리"
            }
        ]
        
        for doc in trading_knowledge:
            embedding = self.embedder.encode(doc['content'])
            self.store.add_vectors(
                collection_name=knowledge_collection,
                ids=[doc['id']],
                embeddings=[embedding],
                documents=[doc['content']],
                metadatas={
                    "title": doc['title'],
                    "category": doc['category']
                }
            )
        
        print(f"  ✅ {len(trading_knowledge)}개 투자 지식 저장 완료")
        print()
        
        # ================================================================
        # Collection 2: 시장 뉴스
        # ================================================================
        print("[2/4] 시장 뉴스 적재 중...")
        
        news_collection = f"market_news_{self.timestamp}"
        self.store.create_collection(news_collection, dimension=384)
        
        market_news = [
            {
                "id": "news_001",
                "title": "삼성전자 3분기 실적 호조",
                "content": """
삼성전자가 3분기 실적 개선을 발표했습니다.
- 매출: 73조원 (전년비 +12%)
- 영업이익: 10.3조원 (전년비 +274%)
메모리 반도체 가격 상승과 D램 수요 증가가 주요 원인입니다.
4분기에도 실적 개선이 지속될 전망입니다.
                """,
                "company": "삼성전자",
                "date": "2024-10-01",
                "sentiment": "긍정"
            },
            {
                "id": "news_002",
                "title": "미국 연준 금리 동결",
                "content": """
미국 연방준비제도가 기준금리를 5.5%로 동결했습니다.
인플레이션이 안정화 추세를 보이고 있어 추가 인상은 없을 것으로 전망됩니다.
증시는 혼조세를 보이고 있으며, 반도체 업종이 강세입니다.
                """,
                "company": "전체시장",
                "date": "2024-10-01",
                "sentiment": "중립"
            },
            {
                "id": "news_003",
                "title": "SK하이닉스 HBM 수주 확대",
                "content": """
SK하이닉스가 AI용 고대역폭메모리(HBM) 수주를 대폭 확대했습니다.
엔비디아向 HBM3 공급이 증가하고 있으며,
4분기에도 공급 부족 현상이 지속될 것으로 보입니다.
                """,
                "company": "SK하이닉스",
                "date": "2024-10-02",
                "sentiment": "긍정"
            }
        ]
        
        for news in market_news:
            embedding = self.embedder.encode(news['content'])
            self.store.add_vectors(
                collection_name=news_collection,
                ids=[news['id']],
                embeddings=[embedding],
                documents=[news['content']],
                metadatas={
                    "title": news['title'],
                    "company": news['company'],
                    "date": news['date'],
                    "sentiment": news['sentiment']
                }
            )
        
        print(f"  ✅ {len(market_news)}개 뉴스 저장 완료")
        print()
        
        # ================================================================
        # Collection 3: 재무 데이터
        # ================================================================
        print("[3/4] 재무 데이터 적재 중...")
        
        financial_collection = f"financial_data_{self.timestamp}"
        self.store.create_collection(financial_collection, dimension=384)
        
        financial_data = [
            {
                "id": "samsung_financial",
                "ticker": "005930",
                "company": "삼성전자",
                "content": """
삼성전자 2024년 3분기 재무 데이터:
- 매출: 73조원
- 영업이익: 10.3조원
- 순이익: 9.1조원
- PER: 15.2 (업종평균 18.5)
- PBR: 1.3
- ROE: 8.5%
- 부채비율: 35%
현재 주가는 내재가치 대비 저평가 상태로 판단됩니다.
                """,
                "quarter": "2024Q3"
            },
            {
                "id": "hynix_financial",
                "ticker": "000660",
                "company": "SK하이닉스",
                "content": """
SK하이닉스 2024년 3분기 재무 데이터:
- 매출: 16.4조원
- 영업이익: 5.8조원
- 순이익: 4.9조원
- PER: 22.5
- PBR: 2.1
- ROE: 12.3%
HBM 수주 증가로 수익성이 크게 개선되었습니다.
                """,
                "quarter": "2024Q3"
            }
        ]
        
        for data in financial_data:
            embedding = self.embedder.encode(data['content'])
            self.store.add_vectors(
                collection_name=financial_collection,
                ids=[data['id']],
                embeddings=[embedding],
                documents=[data['content']],
                metadatas={
                    "ticker": data['ticker'],
                    "company": data['company'],
                    "quarter": data['quarter']
                }
            )
        
        print(f"  ✅ {len(financial_data)}개 재무 데이터 저장 완료")
        print()
        
        # ================================================================
        # Collection 4: 거래 기록
        # ================================================================
        print("[4/4] 거래 기록 적재 중...")
        
        trade_collection = f"trade_history_{self.timestamp}"
        self.store.create_collection(trade_collection, dimension=384)
        
        trade_history = [
            {
                "id": "trade_001",
                "content": """
2024-09-15 삼성전자 매수
- 매수가: 68,000원
- 수량: 100주
- 이유: PER 저평가, 실적 개선 전망
결과: +5% 수익 (70,400원 매도)
교훈: 기술적 지표와 재무 분석 병행이 효과적
                """,
                "ticker": "005930",
                "result": "성공"
            },
            {
                "id": "trade_002",
                "content": """
2024-09-20 네이버 매수
- 매수가: 210,000원
- 수량: 50주
- 이유: AI 서비스 기대감
결과: -3% 손절 (203,700원 매도)
교훈: 테마주 단기 매매는 손절 철저히
                """,
                "ticker": "035420",
                "result": "실패"
            }
        ]
        
        for trade in trade_history:
            embedding = self.embedder.encode(trade['content'])
            self.store.add_vectors(
                collection_name=trade_collection,
                ids=[trade['id']],
                embeddings=[embedding],
                documents=[trade['content']],
                metadatas={
                    "ticker": trade['ticker'],
                    "result": trade['result']
                }
            )
        
        print(f"  ✅ {len(trade_history)}개 거래 기록 저장 완료")
        print()
        
        # 저장된 컬렉션 이름들
        self.collections = {
            "knowledge": knowledge_collection,
            "news": news_collection,
            "financial": financial_collection,
            "trade": trade_collection
        }
        
        print("=" * 80)
        print("✅ 지식 베이스 구축 완료!")
        print("=" * 80)
        print()
    
    def analyze_stock(self, ticker: str, company: str, question: str) -> str:
        """주식 분석"""
        
        print("=" * 80)
        print(f"🤖 AI 에이전트 분석: {company} ({ticker})")
        print("=" * 80)
        print()
        print(f"📝 질문: {question}")
        print()
        
        # ================================================================
        # STEP 1: RAG 검색
        # ================================================================
        print("-" * 80)
        print("STEP 1: 관련 정보 검색 중...")
        print("-" * 80)
        print()
        
        query = f"{company} {question}"
        query_embedding = self.embedder.encode(query)
        
        # 1-1. 재무 데이터 검색
        print("[1] 재무 데이터 검색...")
        financial_results = self.store.search_vectors(
            collection_name=self.collections['financial'],
            query_embedding=query_embedding,
            limit=2
        )
        print(f"  ✅ {len(financial_results)}개 재무 데이터 발견")
        
        # 1-2. 뉴스 검색
        print("[2] 최근 뉴스 검색...")
        news_results = self.store.search_vectors(
            collection_name=self.collections['news'],
            query_embedding=query_embedding,
            limit=3
        )
        print(f"  ✅ {len(news_results)}개 뉴스 발견")
        
        # 1-3. 투자 지식 검색
        print("[3] 관련 투자 지식 검색...")
        knowledge_results = self.store.search_vectors(
            collection_name=self.collections['knowledge'],
            query_embedding=query_embedding,
            limit=2
        )
        print(f"  ✅ {len(knowledge_results)}개 지식 발견")
        
        # 1-4. 과거 거래 기록 검색
        print("[4] 과거 거래 기록 검색...")
        trade_results = self.store.search_vectors(
            collection_name=self.collections['trade'],
            query_embedding=query_embedding,
            limit=2
        )
        print(f"  ✅ {len(trade_results)}개 거래 기록 발견")
        print()
        
        # ================================================================
        # STEP 2: 검색 결과 출력
        # ================================================================
        print("-" * 80)
        print("STEP 2: 검색된 정보")
        print("-" * 80)
        print()
        
        print("📊 [재무 데이터]")
        for i, result in enumerate(financial_results, 1):
            print(f"  [{i}] {result['metadata'].get('company', 'N/A')}")
            print(f"      유사도: {result['similarity_score']:.4f}")
            print(f"      {result['content'][:100]}...")
        print()
        
        print("📰 [최근 뉴스]")
        for i, result in enumerate(news_results, 1):
            print(f"  [{i}] {result['metadata'].get('title', 'N/A')}")
            print(f"      유사도: {result['similarity_score']:.4f}")
            print(f"      {result['content'][:100]}...")
        print()
        
        print("📚 [참고 지식]")
        for i, result in enumerate(knowledge_results, 1):
            print(f"  [{i}] {result['metadata'].get('title', 'N/A')}")
            print(f"      유사도: {result['similarity_score']:.4f}")
        print()
        
        # ================================================================
        # STEP 3: 컨텍스트 구성
        # ================================================================
        print("-" * 80)
        print("STEP 3: LLM 프롬프트 생성 (RAG → LLM 연결!)")
        print("-" * 80)
        print()
        
        # 컨텍스트 구성
        context_parts = []
        
        context_parts.append("[재무 데이터]")
        for result in financial_results:
            context_parts.append(result['content'])
        
        context_parts.append("\n[최근 뉴스]")
        for result in news_results:
            context_parts.append(f"- {result['metadata'].get('title', '')}: {result['content']}")
        
        context_parts.append("\n[참고 투자 지식]")
        for result in knowledge_results:
            context_parts.append(f"- {result['metadata'].get('title', '')}: {result['content']}")
        
        if trade_results:
            context_parts.append("\n[과거 거래 경험]")
            for result in trade_results:
                context_parts.append(result['content'])
        
        context = "\n\n".join(context_parts)
        
        # 프롬프트 생성
        prompt = f"""당신은 전문 주식 애널리스트 AI입니다.

다음 정보를 기반으로 사용자의 질문에 답변해주세요:

{context}

[사용자 질문]
{company} ({ticker}): {question}

[답변 형식]
1. 현재 상황 요약
2. 재무 분석
3. 시장 동향 및 뉴스
4. 투자 의견 (매수/보유/매도)
5. 리스크 요인
6. 구체적 실행 전략

답변:"""
        
        print("생성된 프롬프트:")
        print("┌" + "─" * 78 + "┐")
        print(prompt[:500] + "...")
        print("└" + "─" * 78 + "┘")
        print()
        print("💡 이 프롬프트가 LLM (OpenAI/Claude/Ollama)에 전달됩니다!")
        print()
        
        # ================================================================
        # STEP 4: LLM 답변 (시뮬레이션)
        # ================================================================
        print("-" * 80)
        print("STEP 4: AI 분석 결과")
        print("-" * 80)
        print()
        
        # 실제로는 LLM API 호출
        # answer = openai.ChatCompletion.create(...)
        
        answer = f"""
🤖 AI 애널리스트 분석

1️⃣ 현재 상황 요약
{company}는 2024년 3분기 실적이 크게 개선되었습니다.
재무 데이터에 따르면 영업이익이 전년 대비 큰 폭으로 증가했으며,
최근 뉴스에서도 긍정적인 전망이 나오고 있습니다.

2️⃣ 재무 분석
- PER 15.2: 업종 평균(18.5) 대비 저평가
- PBR 1.3: 적정 수준
- ROE 8.5%: 개선 추세
- 부채비율 35%: 안정적

밸류에이션 측면에서 매력적인 구간입니다.

3️⃣ 시장 동향 및 뉴스
최근 뉴스 분석:
✅ 3분기 실적 호조 발표
✅ 메모리 반도체 가격 상승
✅ D램 수요 증가
✅ 4분기 실적 개선 전망

시장 센티먼트: 긍정적

4️⃣ 투자 의견: 매수 (BUY) ⭐⭐⭐⭐

근거:
- 실적 턴어라운드 확인
- 밸류에이션 매력적 (PER 저평가)
- 업황 개선 사이클 초입
- 과거 유사 상황에서 수익 실현 경험

5️⃣ 리스크 요인
⚠️  주의사항:
- 중국 경기 둔화 우려
- 메모리 가격 변동성
- 환율 리스크
- 지정학적 리스크

6️⃣ 구체적 실행 전략

매수 전략:
- 분할 매수 추천 (3회 나눠서)
  1차: 현재 가격의 40%
  2차: -3% 하락 시 30%
  3차: -5% 하락 시 30%

리스크 관리:
- 손절가: -8%
- 1차 목표가: +15%
- 2차 목표가: +25%
- 트레일링 스탑: 고점 대비 -5%

포지션 크기:
- 전체 자산의 10% 이하 권장
- 과거 거래 기록을 볼 때 분산투자가 효과적

타이밍:
- 단기: 1-3개월
- 중기: 3-6개월
- 실적 발표 전후 변동성 주의

체크포인트:
□ 매수 후 주간 단위로 뉴스 모니터링
□ 월간 단위로 재무 데이터 확인
□ RSI, MACD 등 기술적 지표 병행
□ 손절가 도달 시 감정 배제하고 실행

───────────────────────────────────────

💡 결론: 현재 시점에서 {company}는 매수 적기로 판단됩니다.
        다만 리스크 관리를 철저히 하시기 바랍니다.
        """
        
        print(answer)
        print()
        
        return answer
    
    def cleanup(self):
        """테스트 데이터 정리"""
        print("-" * 80)
        response = input("테스트 데이터를 삭제하시겠습니까? (y/n): ")
        
        if response.lower() == 'y':
            for name, collection in self.collections.items():
                self.store.delete_collection(collection)
                print(f"  ✅ {name} 컬렉션 삭제 완료")
            print("\n✅ 모든 테스트 데이터 삭제 완료")
        else:
            print("\nℹ️  데이터가 유지됩니다.")
    
    def show_util_demo(self):
        """ChromaUtil 기능 데모"""
        print("\n" + "=" * 80)
        print("🔧 ChromaUtil 기능 데모")
        print("=" * 80)
        print()
        
        print("=" * 80)
        print("1️⃣  모든 콜렉션 보기")
        print("=" * 80)
        self.chroma_util.show_collections()
        
        print("=" * 80)
        print("2️⃣  특정 콜렉션의 문서 보기 (0~3개)")
        print("=" * 80)
        results = self.chroma_util.show_documents(
            self.collections['knowledge'], 0, 3
        )
        
        print("=" * 80)
        print("3️⃣  유사도 검색")
        print("=" * 80)
        search_results = self.chroma_util.search_similar(
            self.collections['news'],
            "반도체 실적",
            limit=3
        )
        
        print("=" * 80)
        print("4️⃣  체이닝으로 유사도 필터링 (0.3 이상)")
        print("=" * 80)
        filtered = search_results.get_similarity_gte(0.3)
        print(f"\n필터링 결과: {len(filtered)}개 문서")
        for i, doc in enumerate(filtered, 1):
            print(f"  [{i}] {doc.id}: 유사도 {doc.similarity_score:.4f}")
        
        print("\n" + "=" * 80)
        print("5️⃣  콜렉션 정보 조회")
        print("=" * 80)
        self.chroma_util.get_collection_info(self.collections['financial'])
        
        print("💡 ChromaUtil을 사용하면 ChromaDB를 쉽게 탐색할 수 있습니다!")
        print()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   📈 AI 주식 트레이딩 에이전트 데모                            ║
║                                                                              ║
║  RAG (Vector Search) + LLM을 활용한 실전 주식 분석 시스템                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 에이전트 초기화
    agent = StockAgent()
    
    # 지식 베이스 구축
    agent.setup_knowledge_base()
    
    input("Enter를 눌러 주식 분석 시작...")
    print("\n")
    
    # 주식 분석 실행
    agent.analyze_stock(
        ticker="005930",
        company="삼성전자",
        question="지금 매수해도 될까요? 투자 전략을 알려주세요."
    )
    
    # ChromaUtil 데모
    input("\nEnter를 눌러 ChromaUtil 기능 데모 보기...")
    agent.show_util_demo()
    
    # 정리
    agent.cleanup()
    
    print()
    print("=" * 80)
    print("🎉 데모 완료!")
    print("=" * 80)
    print()
    print("💡 핵심 포인트:")
    print("  1. RAG에 투자 지식, 뉴스, 재무 데이터 저장")
    print("  2. 질문 시 관련 정보를 Vector 검색")
    print("  3. 검색 결과를 LLM 프롬프트로 구성")
    print("  4. LLM이 종합 분석 및 투자 조언 생성")
    print()
    print("📚 더 알아보기:")
    print("  - docs/STOCK_AGENT_RAG_DESIGN.md")
    print("  - docs/RAG_LLM_CONNECTION.md")
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
