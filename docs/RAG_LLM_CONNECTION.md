# 🔗 RAG와 LLM의 연결 원리

Vector 검색으로 찾은 문서가 어떻게 LLM에 전달되는지 자세히 설명합니다.

---

## 🎯 핵심 개념

```
RAG = Retrieval-Augmented Generation
       검색으로 보강된    생성

= Vector 검색 + LLM 생성
  ─────────────  ────────
  관련 문서 찾기   답변 생성
```

---

## 🔄 전체 흐름도

```
┌────────────────────────────────────────────────────────────┐
│                    RAG 시스템 전체 흐름                       │
└────────────────────────────────────────────────────────────┘

사용자 질문: "Python으로 리스트 정렬하는 방법?"
    │
    ↓
┌───┴────────────────────────────────────────────────────┐
│ 1️⃣  RETRIEVAL (검색) - ChromaDB                        │
└────────────────────────────────────────────────────────┘
    │
    │ [질문을 벡터로 변환]
    │ "Python 리스트 정렬" → [0.1, 0.5, 0.8, ...]
    │
    ↓
    │ [유사도 검색]
    │ ChromaDB에서 관련 문서 찾기
    │
    ↓
    │ [검색 결과]
    │ ✅ 문서1: "Python sort() 메서드" (유사도 0.95)
    │ ✅ 문서2: "sorted() 함수 사용법" (유사도 0.92)
    │ ✅ 문서3: "리스트 정렬 예제" (유사도 0.88)
    │
    ↓
┌───┴────────────────────────────────────────────────────┐
│ 2️⃣  AUGMENTATION (보강) - 프롬프트 구성                 │
└────────────────────────────────────────────────────────┘
    │
    │ [컨텍스트 구성]
    │ 검색된 문서들을 하나의 텍스트로 합침:
    │
    │ """
    │ [문서1] Python의 sort() 메서드는 리스트를 
    │        제자리에서 정렬합니다...
    │ 
    │ [문서2] sorted() 함수는 새로운 정렬된 
    │        리스트를 반환합니다...
    │ 
    │ [문서3] 예제 코드:
    │        numbers = [3, 1, 4, 1, 5]
    │        numbers.sort()
    │ """
    │
    ↓
    │ [프롬프트 생성]
    │ 질문 + 컨텍스트를 LLM용 프롬프트로 만듦:
    │
    │ """
    │ 다음 문서를 참고하여 질문에 답변해주세요:
    │
    │ [참고 문서]
    │ [문서1] Python의 sort()...
    │ [문서2] sorted() 함수는...
    │ [문서3] 예제 코드...
    │
    │ [질문]
    │ Python으로 리스트 정렬하는 방법?
    │ """
    │
    ↓
┌───┴────────────────────────────────────────────────────┐
│ 3️⃣  GENERATION (생성) - LLM                            │
└────────────────────────────────────────────────────────┘
    │
    │ [LLM 호출]
    │ OpenAI API, Claude API, Ollama 등에 전달
    │
    ↓
    │ [LLM이 프롬프트 읽음]
    │ - 참고 문서 이해
    │ - 질문 이해
    │ - 문서 기반으로 답변 생성
    │
    ↓
    │ [LLM 답변]
    │ "Python에서 리스트를 정렬하는 방법은 2가지입니다:
    │  
    │  1. sort() 메서드: 원본 리스트를 직접 정렬
    │     numbers = [3, 1, 4]
    │     numbers.sort()
    │  
    │  2. sorted() 함수: 새 정렬된 리스트 반환
    │     new_numbers = sorted([3, 1, 4])
    │  
    │  (참고: 문서1, 문서2)"
    │
    ↓
사용자에게 답변 전달
```

---

## 💻 실제 코드로 보는 연결

### 전체 과정

```python
from rag.vector_store import ChromaVectorStore
from embedding_service import EmbeddingService
from llm_service import LLMService  # OpenAI, Claude, Ollama 등

def rag_answer(user_question: str) -> str:
    """RAG 시스템으로 질문에 답변"""
    
    # ─────────────────────────────────────────────────
    # STEP 1: RETRIEVAL (검색)
    # ─────────────────────────────────────────────────
    
    # 1-1. 초기화
    store = ChromaVectorStore("./chroma-data")
    embedder = EmbeddingService()
    
    # 1-2. 질문을 벡터로 변환
    question_vector = embedder.encode(user_question)
    # "Python 리스트 정렬" → [0.1, 0.5, 0.8, ..., 0.3]
    
    # 1-3. ChromaDB에서 유사 문서 검색
    search_results = store.search_vectors(
        collection_name="tech_docs",
        query_embedding=question_vector,
        limit=3  # 상위 3개 문서
    )
    
    # 검색 결과:
    # [
    #   {"content": "Python sort() 메서드...", "similarity": 0.95},
    #   {"content": "sorted() 함수...", "similarity": 0.92},
    #   {"content": "정렬 예제...", "similarity": 0.88}
    # ]
    
    
    # ─────────────────────────────────────────────────
    # STEP 2: AUGMENTATION (보강) - 핵심 연결 부분!
    # ─────────────────────────────────────────────────
    
    # 2-1. 검색된 문서들을 하나의 컨텍스트로 합치기
    context_parts = []
    for i, result in enumerate(search_results, 1):
        doc_text = result['content']
        context_parts.append(f"[문서{i}] {doc_text}")
    
    context = "\n\n".join(context_parts)
    
    # context 내용:
    # """
    # [문서1] Python의 sort() 메서드는...
    # 
    # [문서2] sorted() 함수는...
    # 
    # [문서3] 정렬 예제...
    # """
    
    
    # 2-2. LLM용 프롬프트 생성 ← 여기가 핵심 연결!
    prompt = f"""당신은 친절한 프로그래밍 도우미입니다.
다음 참고 문서를 기반으로 사용자의 질문에 답변해주세요.

[참고 문서]
{context}

[사용자 질문]
{user_question}

[답변 규칙]
1. 참고 문서의 내용을 기반으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 코드 예제가 있으면 포함하세요
4. 친절하고 명확하게 설명하세요

답변:"""
    
    
    # ─────────────────────────────────────────────────
    # STEP 3: GENERATION (생성)
    # ─────────────────────────────────────────────────
    
    # 3-1. LLM에 프롬프트 전달
    llm = LLMService()
    answer = llm.generate(prompt)
    
    # LLM이 받는 내용:
    # - 참고 문서 3개 (검색된 내용)
    # - 사용자 질문
    # - 답변 규칙
    
    # LLM이 하는 일:
    # - 문서 읽고 이해
    # - 질문에 맞는 답변 생성
    # - 문서 기반으로 정확하게 답변
    
    
    # 3-2. 답변 반환
    return answer

# ═════════════════════════════════════════════════════════
# 사용 예시
# ═════════════════════════════════════════════════════════

question = "Python으로 리스트 정렬하는 방법?"
answer = rag_answer(question)

print(answer)
# 출력:
# """
# Python에서 리스트를 정렬하는 방법은 2가지입니다:
# 
# 1. sort() 메서드 (제자리 정렬)
#    numbers = [3, 1, 4, 1, 5]
#    numbers.sort()
#    print(numbers)  # [1, 1, 3, 4, 5]
# 
# 2. sorted() 함수 (새 리스트 반환)
#    numbers = [3, 1, 4, 1, 5]
#    sorted_numbers = sorted(numbers)
#    print(sorted_numbers)  # [1, 1, 3, 4, 5]
# 
# 주요 차이점: sort()는 원본을 변경하고, sorted()는 
# 새 리스트를 반환합니다.
# """
```

---

## 🔍 상세 분석: 연결 과정

### 1️⃣ 검색 결과 → 컨텍스트 변환

```python
# ChromaDB 검색 결과 (리스트)
search_results = [
    {
        "id": "doc_001",
        "content": "Python의 sort() 메서드는 리스트를 제자리에서 정렬합니다.",
        "similarity_score": 0.95,
        "metadata": {"title": "sort() 사용법"}
    },
    {
        "id": "doc_002", 
        "content": "sorted() 함수는 새로운 정렬된 리스트를 반환합니다.",
        "similarity_score": 0.92,
        "metadata": {"title": "sorted() 함수"}
    }
]

# ↓ 변환 ↓

# LLM용 컨텍스트 (문자열)
context = """
[문서1: sort() 사용법]
Python의 sort() 메서드는 리스트를 제자리에서 정렬합니다.

[문서2: sorted() 함수]
sorted() 함수는 새로운 정렬된 리스트를 반환합니다.
"""
```

### 2️⃣ 컨텍스트 + 질문 → 프롬프트

```python
# 컨텍스트 (검색된 문서)
context = "..."

# 사용자 질문
question = "Python으로 리스트 정렬하는 방법?"

# ↓ 결합 ↓

# LLM 프롬프트 (최종)
prompt = f"""
당신은 프로그래밍 도우미입니다.
다음 문서를 참고하여 질문에 답변하세요:

{context}

질문: {question}
"""

# 이 prompt를 LLM API에 전달!
```

### 3️⃣ 프롬프트 → LLM → 답변

```python
# OpenAI API 예시
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "당신은 프로그래밍 도우미입니다."
        },
        {
            "role": "user",
            "content": prompt  # ← 여기에 컨텍스트 + 질문!
        }
    ]
)

answer = response.choices[0].message.content
```

---

## 🎨 프롬프트 구성 방식 비교

### 방식 A: 간단한 버전

```python
def simple_rag_prompt(context: str, question: str) -> str:
    return f"""
참고 문서:
{context}

질문: {question}

답변:"""
```

### 방식 B: 상세한 버전 (권장)

```python
def detailed_rag_prompt(context: str, question: str) -> str:
    return f"""당신은 전문적인 기술 지원 AI입니다.

[역할]
- 사용자의 기술적 질문에 정확히 답변합니다
- 제공된 문서의 내용만 사용합니다
- 문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답합니다

[참고 문서]
{context}

[사용자 질문]
{question}

[답변 형식]
1. 핵심 답변을 먼저 제시하세요
2. 필요시 코드 예제를 포함하세요
3. 단계별로 설명하세요
4. 참고한 문서를 명시하세요

답변:"""
```

### 방식 C: 메타데이터 포함 버전

```python
def metadata_rag_prompt(results: list, question: str) -> str:
    # 문서에 메타데이터 포함
    context_parts = []
    for i, result in enumerate(results, 1):
        title = result['metadata'].get('title', '제목 없음')
        author = result['metadata'].get('author', '작성자 미상')
        date = result['metadata'].get('date', '날짜 미상')
        content = result['content']
        
        context_parts.append(f"""
[문서 {i}]
제목: {title}
작성자: {author}
날짜: {date}

내용:
{content}
        """)
    
    context = "\n\n".join(context_parts)
    
    return f"""
다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""
```

---

## 🔗 다양한 연결 패턴

### 패턴 1: 기본 RAG (단순 연결)

```
질문 → 검색 → 컨텍스트 → LLM → 답변
```

```python
def basic_rag(question):
    # 1. 검색
    docs = vector_search(question)
    
    # 2. 컨텍스트 생성
    context = "\n".join([d['content'] for d in docs])
    
    # 3. LLM 호출
    prompt = f"문서: {context}\n질문: {question}"
    answer = llm.generate(prompt)
    
    return answer
```

### 패턴 2: 필터링 RAG

```
질문 → 의도 파악 → 필터링 검색 → 컨텍스트 → LLM → 답변
```

```python
def filtered_rag(question):
    # 1. 질문 의도 파악
    category = classify_question(question)
    # "Python 질문" → category="programming"
    
    # 2. 필터링 검색
    docs = vector_search(
        question,
        where={"category": category}  # 필터!
    )
    
    # 3. 컨텍스트 생성
    context = "\n".join([d['content'] for d in docs])
    
    # 4. LLM 호출
    prompt = f"문서: {context}\n질문: {question}"
    answer = llm.generate(prompt)
    
    return answer
```

### 패턴 3: 다단계 RAG (Re-ranking)

```
질문 → 검색 (10개) → 재순위 (3개) → 컨텍스트 → LLM → 답변
```

```python
def reranking_rag(question):
    # 1. 1차 검색 (많이)
    docs = vector_search(question, limit=10)
    
    # 2. 재순위 (질문과 가장 관련성 높은 것만)
    reranked = rerank(question, docs)
    top_docs = reranked[:3]
    
    # 3. 컨텍스트 생성
    context = "\n".join([d['content'] for d in top_docs])
    
    # 4. LLM 호출
    prompt = f"문서: {context}\n질문: {question}"
    answer = llm.generate(prompt)
    
    return answer
```

### 패턴 4: 하이브리드 RAG (키워드 + 벡터)

```
질문 → 키워드 검색 + 벡터 검색 → 결합 → 컨텍스트 → LLM → 답변
```

```python
def hybrid_rag(question):
    # 1-1. 벡터 검색
    vector_docs = vector_search(question, limit=5)
    
    # 1-2. 키워드 검색
    keywords = extract_keywords(question)
    keyword_docs = keyword_search(keywords, limit=5)
    
    # 2. 결합 및 중복 제거
    all_docs = merge_and_dedupe(vector_docs, keyword_docs)
    
    # 3. 컨텍스트 생성
    context = "\n".join([d['content'] for d in all_docs])
    
    # 4. LLM 호출
    prompt = f"문서: {context}\n질문: {question}"
    answer = llm.generate(prompt)
    
    return answer
```

---

## 💡 RAG vs LLM 단독 사용

### LLM만 사용 (RAG 없이)

```python
def llm_only(question):
    """RAG 없이 LLM만 사용"""
    prompt = f"질문: {question}\n답변:"
    answer = llm.generate(prompt)
    return answer

# 문제점:
# ❌ LLM이 모르는 내용은 답변 못 함
# ❌ 최신 정보 없음 (학습 데이터까지만)
# ❌ 회사 내부 정보 모름
# ❌ 환각(Hallucination) 가능
```

### RAG + LLM 사용

```python
def rag_with_llm(question):
    """RAG로 문서 찾고 LLM이 답변"""
    
    # 1. 관련 문서 검색
    docs = vector_search(question)
    context = "\n".join([d['content'] for d in docs])
    
    # 2. 문서 + 질문을 LLM에 전달
    prompt = f"문서: {context}\n질문: {question}"
    answer = llm.generate(prompt)
    
    return answer

# 장점:
# ✅ LLM이 문서 기반으로 정확히 답변
# ✅ 최신 정보 반영 (문서만 업데이트하면 됨)
# ✅ 회사 내부 정보 활용 가능
# ✅ 환각 크게 감소 (문서 기반)
```

### 비교

```
질문: "우리 회사 2024년 3분기 매출은?"

[LLM만 사용]
LLM: "죄송합니다. 회사의 최신 재무 정보는 
      제 학습 데이터에 없습니다."
❌ 답변 불가

[RAG + LLM]
1. ChromaDB 검색 → "2024 Q3 실적 보고서" 찾음
2. LLM에 보고서 + 질문 전달
3. LLM: "2024년 3분기 매출은 1,250억원입니다.
        전년 대비 15% 증가했습니다."
✅ 정확한 답변!
```

---

## 🎯 핵심 정리

### RAG와 LLM의 역할 분담

```
┌─────────────────────────────────────────┐
│ RAG (Vector Search)                     │
│ ────────────────────────────────────── │
│ 역할: 관련 문서 찾기                     │
│ 입력: 사용자 질문                        │
│ 출력: 관련 문서 3-5개                    │
│ 특징: 빠르고 정확한 검색                 │
└─────────────────────────────────────────┘
              ↓
        (문서 전달)
              ↓
┌─────────────────────────────────────────┐
│ LLM (Language Model)                    │
│ ────────────────────────────────────── │
│ 역할: 문서 기반 답변 생성                │
│ 입력: 문서 + 질문                        │
│ 출력: 자연스러운 답변                    │
│ 특징: 이해력, 생성력                     │
└─────────────────────────────────────────┘
```

### 연결 핵심 코드

```python
# 1. 검색 (RAG)
docs = chromadb.search(question)  # 문서 찾기

# 2. 연결 (Augmentation)
context = combine(docs)  # 문서들 합치기
prompt = f"{context}\n{question}"  # 프롬프트 만들기

# 3. 생성 (LLM)
answer = llm.generate(prompt)  # LLM이 답변 생성
```

---

**핵심: RAG는 관련 문서를 찾아서 LLM에게 "참고 자료"로 전달합니다!** 📚→🤖
