# 📈 주식 트레이딩 에이전트를 위한 RAG 설계

주식 에이전트에서 RAG를 어떻게 활용하는지 실전 예시로 설명합니다.

---

## 🎯 핵심 아이디어

```
주식 에이전트 = LLM + RAG (지식 베이스)
                      │
                      ├─ 기술적 분석 지식
                      ├─ 투자 전략
                      ├─ 시장 뉴스
                      ├─ 재무 데이터
                      └─ 과거 거래 기록
```

**왜 RAG가 필요한가?**
- ✅ LLM은 최신 주가 정보를 모름
- ✅ LLM은 회사별 재무제표를 모름
- ✅ LLM은 오늘의 뉴스를 모름
- ✅ LLM은 당신의 투자 전략을 모름

**해결: RAG로 실시간 정보 제공!**

---

## 📚 주식 에이전트 RAG 구성

### 1. Collection 설계

```python
# Collection 1: 투자 지식 (정적 - 가끔 업데이트)
collection_knowledge = "trading_knowledge"
- 기술적 분석 용어 (RSI, MACD, 볼린저 밴드...)
- 투자 전략 (가치투자, 모멘텀, 스윙...)
- 리스크 관리 기법
- 차트 패턴 (헤드앤숄더, 삼각수렴...)

# Collection 2: 시장 뉴스 (동적 - 실시간 업데이트)
collection_news = "market_news"
- 최근 경제 뉴스
- 기업 공시
- 증권사 리포트
- 시장 분석 기사

# Collection 3: 재무 데이터 (동적 - 정기 업데이트)
collection_financials = "financial_data"
- 기업별 재무제표
- 실적 발표 내용
- 애널리스트 의견
- 배당 정보

# Collection 4: 과거 거래 (동적 - 매 거래마다)
collection_trades = "trade_history"
- 내 거래 기록
- 손익 분석
- 성공/실패 패턴
- 반성 및 교훈
```

---

## 💻 실제 구현 예시

### 초기 설정 (1회)

```python
# setup_stock_rag.py

from rag.vector_store import ChromaVectorStore
from embedding_service import EmbeddingService

store = ChromaVectorStore("./stock-rag-data")
embedder = EmbeddingService()

# ═══════════════════════════════════════════════════════
# Collection 1: 투자 지식 베이스
# ═══════════════════════════════════════════════════════

store.create_collection("trading_knowledge", dimension=384)

knowledge_docs = [
    {
        "id": "rsi_001",
        "content": """
RSI (Relative Strength Index, 상대강도지수)
- 범위: 0~100
- 70 이상: 과매수 (매도 신호)
- 30 이하: 과매도 (매수 신호)
- 14일 기준이 일반적
        """,
        "metadata": {
            "category": "기술적지표",
            "type": "모멘텀",
            "difficulty": "초급"
        }
    },
    {
        "id": "macd_001",
        "content": """
MACD (Moving Average Convergence Divergence)
- 단기 이동평균선과 장기 이동평균선의 차이
- 골든크로스: 매수 신호
- 데드크로스: 매도 신호
- 시그널선과 함께 사용
        """,
        "metadata": {
            "category": "기술적지표",
            "type": "추세",
            "difficulty": "중급"
        }
    },
    {
        "id": "value_investing",
        "content": """
가치투자 전략
- 내재가치 대비 저평가된 주식 매수
- 주요 지표: PER, PBR, ROE
- 장기 보유 원칙
- 대표 투자자: 워렌 버핏
        """,
        "metadata": {
            "category": "투자전략",
            "type": "장기투자",
            "difficulty": "중급"
        }
    },
    {
        "id": "risk_management",
        "content": """
리스크 관리 원칙
1. 손절매 설정 (보통 -5~10%)
2. 분산투자 (계란을 한 바구니에 담지 말 것)
3. 포지션 크기 조절 (한 종목에 전체 자산의 10% 이하)
4. 감정적 거래 금지
        """,
        "metadata": {
            "category": "리스크관리",
            "type": "기본원칙",
            "difficulty": "필수"
        }
    }
]

# 저장
for doc in knowledge_docs:
    embedding = embedder.encode(doc['content'])
    store.add_vectors(
        collection_name="trading_knowledge",
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['content']],
        metadatas=[doc['metadata']]
    )

print("✅ 투자 지식 베이스 구축 완료")


# ═══════════════════════════════════════════════════════
# Collection 2: 시장 뉴스 (실시간 업데이트)
# ═══════════════════════════════════════════════════════

store.create_collection("market_news", dimension=384)

# 뉴스는 크롤러나 API로 실시간 수집
news_docs = [
    {
        "id": "news_20241001_001",
        "content": """
삼성전자, 3분기 영업이익 10조원 전망
- 메모리 반도체 가격 상승
- D램 수요 증가
- 4분기 실적 개선 기대
        """,
        "metadata": {
            "company": "삼성전자",
            "date": "2024-10-01",
            "category": "실적",
            "sentiment": "긍정"
        }
    },
    {
        "id": "news_20241001_002",
        "content": """
미국 연준, 금리 동결 결정
- 기준금리 5.5% 유지
- 인플레이션 안정화 관찰
- 증시 혼조세 예상
        """,
        "metadata": {
            "region": "미국",
            "date": "2024-10-01",
            "category": "금리",
            "sentiment": "중립"
        }
    }
]

for doc in news_docs:
    embedding = embedder.encode(doc['content'])
    store.add_vectors(
        collection_name="market_news",
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['content']],
        metadatas=[doc['metadata']]
    )

print("✅ 시장 뉴스 초기 적재 완료")


# ═══════════════════════════════════════════════════════
# Collection 3: 재무 데이터
# ═══════════════════════════════════════════════════════

store.create_collection("financial_data", dimension=384)

financial_docs = [
    {
        "id": "samsung_q3_2024",
        "content": """
삼성전자 2024년 3분기 실적
- 매출: 73조원 (전년 대비 +12%)
- 영업이익: 10.3조원 (전년 대비 +274%)
- 순이익: 9.1조원
- PER: 15.2
- PBR: 1.3
- ROE: 8.5%
        """,
        "metadata": {
            "company": "삼성전자",
            "ticker": "005930",
            "quarter": "2024Q3",
            "type": "실적"
        }
    }
]

for doc in financial_docs:
    embedding = embedder.encode(doc['content'])
    store.add_vectors(
        collection_name="financial_data",
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['content']],
        metadatas=[doc['metadata']]
    )

print("✅ 재무 데이터 초기 적재 완료")
```

---

## 🔄 실시간 업데이트 (자동화)

### 1. 뉴스 크롤러 (매시간 실행)

```python
# news_crawler.py

import schedule
import time
from datetime import datetime

def fetch_and_store_news():
    """뉴스 크롤링 및 저장"""
    
    # 1. 뉴스 수집 (API 또는 크롤링)
    news_list = fetch_latest_news()  # 네이버, 다음, 증권사 등
    
    # 2. ChromaDB에 저장
    store = ChromaVectorStore("./stock-rag-data")
    embedder = EmbeddingService()
    
    for news in news_list:
        news_id = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{news['id']}"
        
        embedding = embedder.encode(news['content'])
        
        store.add_vectors(
            collection_name="market_news",
            ids=[news_id],
            embeddings=[embedding],
            documents=[news['content']],
            metadatas={
                "title": news['title'],
                "date": news['date'],
                "source": news['source'],
                "company": news['company'],
                "category": news['category']
            }
        )
    
    print(f"✅ {len(news_list)}개 뉴스 업데이트 완료")

# 매시간 실행
schedule.every().hour.do(fetch_and_store_news)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. 재무 데이터 업데이트 (분기마다)

```python
# financial_updater.py

def update_financial_data(ticker: str):
    """실적 발표 시 자동 업데이트"""
    
    # 1. 재무 데이터 수집 (API)
    financial_data = fetch_financial_data(ticker)
    
    # 2. 텍스트로 변환
    content = f"""
{financial_data['company']} {financial_data['quarter']} 실적
- 매출: {financial_data['revenue']}
- 영업이익: {financial_data['operating_profit']}
- 순이익: {financial_data['net_profit']}
- PER: {financial_data['per']}
- PBR: {financial_data['pbr']}
    """
    
    # 3. ChromaDB에 저장
    store = ChromaVectorStore("./stock-rag-data")
    embedder = EmbeddingService()
    
    embedding = embedder.encode(content)
    
    store.add_vectors(
        collection_name="financial_data",
        ids=[f"{ticker}_{financial_data['quarter']}"],
        embeddings=[embedding],
        documents=[content],
        metadatas={
            "ticker": ticker,
            "quarter": financial_data['quarter'],
            "type": "실적"
        }
    )
    
    print(f"✅ {ticker} 재무 데이터 업데이트 완료")
```

---

## 🤖 주식 에이전트 사용 예시

### 시나리오 1: 종목 분석 요청

```python
# stock_agent.py

def analyze_stock(ticker: str, question: str) -> str:
    """주식 분석 에이전트"""
    
    store = ChromaVectorStore("./stock-rag-data")
    embedder = EmbeddingService()
    llm = LLMService()
    
    # ─────────────────────────────────────────────────
    # STEP 1: 관련 정보 검색 (RAG)
    # ─────────────────────────────────────────────────
    
    query = f"{ticker} {question}"
    query_embedding = embedder.encode(query)
    
    # 1-1. 재무 데이터 검색
    financial_results = store.search_vectors(
        collection_name="financial_data",
        query_embedding=query_embedding,
        where={"ticker": ticker},
        limit=3
    )
    
    # 1-2. 관련 뉴스 검색
    news_results = store.search_vectors(
        collection_name="market_news",
        query_embedding=query_embedding,
        where={"company": ticker},
        limit=5
    )
    
    # 1-3. 투자 지식 검색
    knowledge_results = store.search_vectors(
        collection_name="trading_knowledge",
        query_embedding=query_embedding,
        limit=3
    )
    
    # ─────────────────────────────────────────────────
    # STEP 2: 컨텍스트 구성
    # ─────────────────────────────────────────────────
    
    context = f"""
[재무 데이터]
{format_results(financial_results)}

[최근 뉴스]
{format_results(news_results)}

[참고 투자 지식]
{format_results(knowledge_results)}
    """
    
    # ─────────────────────────────────────────────────
    # STEP 3: LLM 프롬프트 생성
    # ─────────────────────────────────────────────────
    
    prompt = f"""당신은 전문 주식 애널리스트입니다.

다음 정보를 기반으로 질문에 답변해주세요:

{context}

[질문]
{ticker} 종목에 대해: {question}

[답변 형식]
1. 현재 상황 요약
2. 재무 분석
3. 시장 동향
4. 투자 의견 (매수/보유/매도)
5. 리스크 요인

답변:"""
    
    # ─────────────────────────────────────────────────
    # STEP 4: LLM 답변 생성
    # ─────────────────────────────────────────────────
    
    answer = llm.generate(prompt)
    
    return answer


# ═════════════════════════════════════════════════════
# 사용 예시
# ═════════════════════════════════════════════════════

question = "삼성전자를 지금 매수해도 될까요?"
analysis = analyze_stock("005930", question)

print(analysis)
```

**출력 예시:**

```
[AI 분석]

1. 현재 상황 요약
삼성전자는 2024년 3분기 실적이 크게 개선되었습니다. 
매출 73조원, 영업이익 10.3조원으로 전년 대비 각각 12%, 274% 증가했습니다.

2. 재무 분석
- PER 15.2: 업종 평균(18.5) 대비 저평가
- PBR 1.3: 안정적 수준
- ROE 8.5%: 개선 추세
- 메모리 반도체 가격 상승으로 수익성 개선

3. 시장 동향
최근 뉴스에 따르면:
- D램 수요 증가세 지속
- 4분기 실적 추가 개선 전망
- 미국 금리 동결로 안정적 환경

4. 투자 의견: 매수
근거:
- 실적 턴어라운드 확인
- 밸류에이션 매력적
- 메모리 업황 개선 사이클 초입

5. 리스크 요인
- 중국 경기 둔화
- 메모리 가격 변동성
- 지정학적 리스크

추천 매수 전략:
- 분할 매수 (3회 나눠서)
- 손절가: -8%
- 목표가: +20%
```

---

## 📊 시나리오별 활용

### 시나리오 2: 기술적 지표 해석

```python
user_question = "삼성전자 차트에서 RSI가 75인데 어떻게 해야 하나요?"

# RAG가 찾아올 정보:
# 1. RSI 지표 설명 (trading_knowledge)
# 2. 삼성전자 최근 차트 데이터 (technical_data)
# 3. 과거 유사 상황 (trade_history)

# LLM 답변:
"""
RSI 75는 과매수 구간입니다.

[참고 지식]
- RSI 70 이상: 과매수, 조정 가능성
- 하락 전환 신호 주의

[삼성전자 현황]
- 최근 급등으로 RSI 상승
- 거래량 증가 확인

[추천 액션]
1. 부분 익절 고려 (30%)
2. 나머지는 트레일링 스탑 설정
3. 추가 매수는 RSI 50 이하 대기

[과거 사례]
2024년 7월에도 RSI 78 기록 후 -5% 조정
"""
```

### 시나리오 3: 포트폴리오 리밸런싱

```python
user_question = "현재 포트폴리오가 삼성전자 40%, SK하이닉스 30%, 네이버 30%인데 조정이 필요한가요?"

# RAG가 찾아올 정보:
# 1. 분산투자 원칙 (trading_knowledge)
# 2. 각 종목의 최근 뉴스 (market_news)
# 3. 상관관계 분석 (portfolio_analysis)

# LLM 답변:
"""
포트폴리오 분석 결과:

[리스크 평가]
⚠️  문제점:
- 반도체 업종 집중도: 70% (삼성+하이닉스)
- 업종 분산 부족

[권장 조정안]
1. 반도체 비중 축소: 70% → 50%
2. 다른 섹터 편입:
   - 2차전지 10%
   - 금융 10%
   - 바이오 10%

[실행 전략]
- 네이버 유지 (30%)
- 삼성/하이닉스 일부 매도
- 신규 섹터 분할 매수

[근거]
리스크 관리 원칙에 따르면, 한 업종에 
50% 이상 집중은 위험합니다.
"""
```

### 시나리오 4: 공포/탐욕 지수 해석

```python
user_question = "오늘 공포탐욕지수가 85인데 무슨 의미인가요?"

# RAG가 찾아올 정보:
# 1. 공포탐욕지수 설명 (trading_knowledge)
# 2. 오늘의 시장 뉴스 (market_news)
# 3. 과거 유사 시점 데이터 (historical_data)

# LLM 답변:
"""
공포탐욕지수 85 = 극도의 탐욕 (Extreme Greed)

[의미]
- 시장 과열 신호
- 투자자들이 지나치게 낙관적
- 조정 위험 증가

[현재 상황]
오늘 뉴스:
- 코스피 3% 상승
- 거래대금 20조원 돌파
- 개인 순매수 급증

[역사적 패턴]
공포탐욕지수 80 이상 시:
- 1개월 내 조정: 75%
- 평균 조정폭: -5~8%

[추천 행동]
1. 신규 매수 자제
2. 부분 익절 고려
3. 현금 비중 확대
4. 조정 시 매수 기회 대기

역발상 전략: "남들이 탐욕적일 때 두려워하라"
"""
```

---

## 🎯 RAG 데이터 업데이트 전략

### 업데이트 주기

```
┌─────────────────────────────────────────┐
│ 데이터 타입          │ 업데이트 주기      │
├─────────────────────────────────────────┤
│ 투자 지식            │ 월 1회            │
│ 기술적 지표 설명      │ 분기 1회          │
│ 투자 전략            │ 필요시            │
├─────────────────────────────────────────┤
│ 시장 뉴스            │ 실시간 (매시간)   │
│ 공시                │ 실시간            │
│ 증권사 리포트        │ 일 1회            │
├─────────────────────────────────────────┤
│ 재무 데이터          │ 분기별            │
│ 주가 데이터          │ 일 1회            │
│ 거래량              │ 일 1회            │
├─────────────────────────────────────────┤
│ 내 거래 기록         │ 거래 시마다       │
│ 포트폴리오          │ 일 1회            │
│ 손익 분석            │ 주 1회            │
└─────────────────────────────────────────┘
```

### 자동화 스크립트

```python
# auto_updater.py

import schedule

# 1. 매시간: 뉴스 업데이트
schedule.every().hour.do(update_news)

# 2. 매일 오전 9시: 주가/재무 데이터
schedule.every().day.at("09:00").do(update_market_data)

# 3. 매주 일요일: 정리 및 최적화
schedule.every().sunday.at("00:00").do(cleanup_old_data)

# 4. 분기별: 재무제표 업데이트
schedule.every(90).days.do(update_financials)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 💡 핵심 정리

### RAG가 주식 에이전트에 필수인 이유

```
LLM 단독:
❌ "삼성전자 주가?" → "모릅니다" (최신 정보 없음)
❌ "실적이 좋나요?" → "모릅니다" (재무 데이터 없음)
❌ "오늘 뉴스는?" → "모릅니다" (실시간 정보 없음)

RAG + LLM:
✅ "삼성전자 주가?" → "70,000원, 전일 대비 +2.5%" (실시간 데이터)
✅ "실적이 좋나요?" → "Q3 영업이익 10조원, 전년비 +274%" (재무 데이터)
✅ "오늘 뉴스는?" → "반도체 가격 상승, D램 수요 증가" (최신 뉴스)
```

### 데이터 구성 전략

```
[정적 데이터] - 천천히 변함
└─ 투자 지식, 기술적 지표, 전략
   → Collection: trading_knowledge
   → 업데이트: 월 1회

[동적 데이터] - 빠르게 변함
└─ 뉴스, 주가, 공시
   → Collection: market_news
   → 업데이트: 실시간 (매시간)

[개인 데이터] - 나만의 데이터
└─ 거래 기록, 포트폴리오, 전략
   → Collection: my_trading
   → 업데이트: 거래 시마다
```

---

**당신이 정확히 이해한 것:**

✅ RAG에 투자 지식을 미리 저장
✅ 뉴스/데이터를 실시간 업데이트
✅ LLM이 이 정보를 활용해 분석
✅ 최신 정보 기반의 정확한 조언

**이것이 바로 실전 AI 트레이딩 에이전트입니다!** 🚀📈
