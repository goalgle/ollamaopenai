# 🔄 RAG 저장 시점과 워크플로우

에이전트에서 RAG에 데이터를 저장하는 시점과 전체 흐름을 설명합니다.

---

## 🎯 핵심 개념: RAG의 2단계

```
[1단계] 저장 (Indexing)
   문서 → 청크 분할 → 임베딩 → ChromaDB 저장
   (사전 작업, 사용자 질문 전)

[2단계] 검색 및 생성 (Retrieval + Generation)
   질문 → 벡터 검색 → 관련 문서 → LLM → 답변
   (사용자 질문 시)
```

**중요:** 저장은 **질문하기 전**에 미리 해둡니다!

---

## 📅 저장 시점 (When to Store)

### 시나리오별 저장 시점

```
시나리오 1: 초기 설정
───────────────────────────────────────
[회사 설립 또는 시스템 구축 시]
1. 기존 문서 수집
2. ChromaDB에 일괄 저장
3. 이제 질문 받을 준비 완료!

예시:
- 회사 위키 전체 (1,000개 문서)
- 제품 매뉴얼 (500개 문서)
- FAQ (200개 문서)
→ 한 번에 저장! (시간: 1-2시간)
```

```
시나리오 2: 실시간 추가
───────────────────────────────────────
[새 문서 생성 시마다]
- 블로그 포스트 발행 → 즉시 저장
- 고객 문의 답변 → 즉시 저장
- 새 제품 출시 → 매뉴얼 저장
- 정책 변경 → 업데이트된 정책 저장

예시:
- 사용자가 "블로그 작성" 버튼 클릭
- 블로그 저장 → ChromaDB에도 자동 저장
```

```
시나리오 3: 주기적 동기화
───────────────────────────────────────
[배치 작업]
- 매일 자정: 새 문서 수집 → 저장
- 매주 일요일: 전체 재색인
- 매월 1일: 오래된 문서 삭제

예시:
- Cron Job: 0 0 * * * python sync_documents.py
```

```
시나리오 4: 사용자 업로드 시
───────────────────────────────────────
[사용자가 파일 업로드]
1. 파일 업로드 (PDF, DOCX, TXT)
2. 텍스트 추출
3. ChromaDB 저장
4. "업로드 완료! 이제 질문할 수 있어요"

예시:
- Notion: 페이지 생성 → 자동 색인
- Google Drive: 파일 추가 → 자동 색인
```

---

## 🔄 전체 워크플로우

### Phase 1: 저장 단계 (사전 작업)

```
┌─────────────────────────────────────────────────┐
│         [저장 단계 - 미리 준비]                  │
└─────────────────────────────────────────────────┘

1. 문서 수집
   📄 회사 위키
   📄 제품 매뉴얼
   📄 FAQ
   📄 이전 고객 문의
   
   ↓

2. 전처리
   - 텍스트 추출 (PDF → Text)
   - 청크 분할 (긴 문서 → 작은 조각)
   - 중복 제거
   
   ↓

3. 임베딩 생성
   각 청크 → 임베딩 모델 → 벡터
   "Python 튜토리얼" → [0.1, 0.5, 0.8, ...]
   
   ↓

4. ChromaDB 저장
   Collection에 저장
   (id, embedding, document, metadata)
   
   ↓

✅ 준비 완료! 이제 질문 받을 수 있음
```

### Phase 2: 질문 단계 (실시간)

```
┌─────────────────────────────────────────────────┐
│      [질문 단계 - 사용자 질문 시]                │
└─────────────────────────────────────────────────┘

사용자: "Python 배우는 방법 알려줘"
   
   ↓

1. 질문을 벡터로 변환
   "Python 배우는 방법" → [0.1, 0.5, 0.8, ...]
   
   ↓

2. ChromaDB 검색 (0.01초)
   유사도 높은 문서 찾기
   → "Python 튜토리얼" (0.95)
   → "Python 입문 가이드" (0.92)
   → "프로그래밍 시작하기" (0.88)
   
   ↓

3. 컨텍스트 구성
   질문 + 관련 문서들
   
   ↓

4. LLM에 전달
   "다음 문서를 참고해서 답변해줘:
    [문서1] Python 튜토리얼...
    [문서2] Python 입문 가이드...
    
    질문: Python 배우는 방법 알려줘"
   
   ↓

5. 답변 생성
   LLM이 문서 기반으로 답변
   
   ↓

✅ 사용자에게 답변 전달
```

---

## 💻 실제 코드 예제

### 예제 1: 초기 설정 (한 번만)

```python
# setup_rag.py - 최초 1회 실행

from rag.vector_store import ChromaVectorStore
from embedding_service import EmbeddingService

# 1. 초기화
store = ChromaVectorStore("./chroma-data")
embedder = EmbeddingService()

# 2. 기존 문서 수집
documents = [
    {"id": "doc_001", "text": "Python은 프로그래밍 언어입니다", "category": "tech"},
    {"id": "doc_002", "text": "환불은 14일 이내 가능합니다", "category": "policy"},
    {"id": "doc_003", "text": "로그인은 이메일로 가능합니다", "category": "faq"},
    # ... 1000개 문서
]

# 3. Collection 생성
store.create_collection("company_knowledge", dimension=384)

# 4. 일괄 저장
print("문서 저장 중...")
for doc in documents:
    embedding = embedder.encode(doc['text'])
    store.add_vectors(
        collection_name="company_knowledge",
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['text']],
        metadatas=[{"category": doc['category']}]
    )

print(f"✅ {len(documents)}개 문서 저장 완료!")
print("이제 사용자 질문을 받을 수 있습니다.")
```

### 예제 2: 챗봇 질문 처리 (매번)

```python
# chatbot.py - 사용자 질문마다 실행

from rag.vector_store import ChromaVectorStore
from embedding_service import EmbeddingService
from llm_service import LLMService

def answer_question(user_question: str) -> str:
    """사용자 질문에 답변"""
    
    # 1. 초기화 (이미 저장된 데이터 사용)
    store = ChromaVectorStore("./chroma-data")
    embedder = EmbeddingService()
    llm = LLMService()
    
    # 2. 질문을 벡터로 변환
    question_embedding = embedder.encode(user_question)
    
    # 3. 관련 문서 검색 (이미 저장된 것에서)
    results = store.search_vectors(
        collection_name="company_knowledge",
        query_embedding=question_embedding,
        limit=3
    )
    
    # 4. 컨텍스트 구성
    context = "\n\n".join([r['content'] for r in results])
    
    # 5. LLM에 전달
    prompt = f"""
    다음 문서를 참고해서 질문에 답변해주세요:
    
    {context}
    
    질문: {user_question}
    """
    
    answer = llm.generate(prompt)
    
    return answer

# 사용
user_input = "환불은 어떻게 하나요?"
response = answer_question(user_input)
print(response)
# → "환불은 구매일로부터 14일 이내에 가능합니다..."
```

### 예제 3: 새 문서 추가 (실시간)

```python
# add_document.py - 새 문서 생성 시

def add_new_document(title: str, content: str, category: str):
    """새 문서를 시스템에 추가"""
    
    # 1. 데이터베이스에 저장 (일반 DB)
    doc_id = database.insert({
        "title": title,
        "content": content,
        "category": category
    })
    
    # 2. RAG에도 저장 (즉시!)
    store = ChromaVectorStore("./chroma-data")
    embedder = EmbeddingService()
    
    embedding = embedder.encode(content)
    store.add_vectors(
        collection_name="company_knowledge",
        ids=[f"doc_{doc_id}"],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"title": title, "category": category}]
    )
    
    print(f"✅ 문서 '{title}' 저장 완료!")
    print("이제 사용자가 이 내용에 대해 질문할 수 있습니다.")

# 사용 예시
add_new_document(
    title="새로운 배송 정책",
    content="2024년 10월부터 모든 주문은 무료 배송됩니다.",
    category="policy"
)
# → 즉시 RAG에 저장!
# → 바로 질문 가능: "배송비는 얼마인가요?"
```

---

## 🏢 실제 서비스 시나리오

### 시나리오 A: 고객 지원 챗봇

```
[저장 시점]
──────────────────────────────────────
1. 초기 설정 (회사 설립 시)
   - 전체 FAQ 저장 (200개)
   - 제품 매뉴얼 저장 (50개)
   - 이용약관 저장 (10개)
   
2. 매일 추가 (실시간)
   - 새 FAQ 추가됨 → 즉시 저장
   - 정책 변경 → 즉시 업데이트
   
3. 매주 정리 (일요일 자정)
   - 오래된 문서 삭제
   - 중복 문서 정리


[사용 시점]
──────────────────────────────────────
고객: "환불 정책이 뭔가요?"
   ↓
[벡터 검색] (0.01초)
   → "환불 정책" 문서 찾음
   ↓
[LLM 답변 생성] (2초)
   → "환불은 14일 이내..."
   ↓
고객에게 답변


타임라인:
─────────────────────────────────────────────
[아침 9시] 
  관리자: 새 배송 정책 추가
  → ChromaDB에 즉시 저장 ✅

[오후 2시]
  고객: "배송은 며칠 걸리나요?"
  → 방금 추가한 정책으로 답변! ✅
```

### 시나리오 B: 회사 내부 위키 검색

```
[저장 시점]
──────────────────────────────────────
1. 초기 마이그레이션
   - Confluence 전체 문서 (5,000개)
   - Google Docs (2,000개)
   - Notion 페이지 (1,000개)
   → 3일 동안 배치 작업으로 저장
   
2. 실시간 동기화
   - 직원이 문서 작성 → 즉시 색인
   - 문서 수정 → 즉시 업데이트
   - 문서 삭제 → 즉시 제거


[사용 시점]
──────────────────────────────────────
직원: "휴가 신청은 어떻게 하나요?"
   ↓
[검색] HR 문서에서 "휴가 신청" 찾음
   ↓
[답변] "휴가 신청은 인사팀 포털에서..."


타임라인:
─────────────────────────────────────────────
[월요일 오전]
  HR팀: 새 휴가 정책 문서 작성
  → 저장 시 자동으로 ChromaDB에 추가 ✅

[월요일 오후]
  직원: "새 휴가 정책이 뭐죠?"
  → 즉시 검색해서 답변 가능! ✅
```

### 시나리오 C: 기술 문서 검색

```
[저장 시점]
──────────────────────────────────────
1. 프로젝트 시작 시
   - API 문서 저장
   - 코딩 가이드라인 저장
   - 아키텍처 문서 저장
   
2. Git 커밋 시
   - README 변경 → 자동 업데이트
   - 새 문서 추가 → 자동 색인
   (Git Hook으로 자동화)


[사용 시점]
──────────────────────────────────────
개발자: "API 인증은 어떻게 하나요?"
   ↓
[검색] API 문서에서 인증 부분 찾음
   ↓
[답변] "API 인증은 JWT 토큰을 사용..."


자동화 예시:
─────────────────────────────────────────────
.git/hooks/post-commit:

#!/bin/bash
# 커밋 후 자동으로 문서 색인
python scripts/index_docs.py

→ 개발자가 문서 수정 후 커밋
→ 자동으로 RAG 업데이트
→ 팀원들이 즉시 최신 정보 검색 가능!
```

---

## ⏱️ 타이밍 비교

### 저장 (Indexing) - 느림 (한 번만)

```
작업: 1,000개 문서 저장
시간: 10-30분
빈도: 초기 1회 또는 배치 작업

과정:
1. 텍스트 추출: 5분
2. 임베딩 생성: 15분 (API 호출)
3. DB 저장: 5분

→ 하지만 한 번만 하면 됨!
```

### 검색 (Retrieval) - 빠름 (매번)

```
작업: 질문에 답변
시간: 0.01~2초
빈도: 사용자 질문마다

과정:
1. 질문 임베딩: 0.1초
2. 벡터 검색: 0.01초 (ChromaDB)
3. LLM 답변: 1-2초

→ 거의 실시간!
```

---

## 📋 체크리스트

### 초기 설정 시 (한 번)

```
□ ChromaDB 설치 및 초기화
□ 임베딩 모델 선택 (OpenAI, HuggingFace 등)
□ Collection 생성
□ 기존 문서 수집
□ 전처리 (청크 분할)
□ 임베딩 생성 및 저장
□ 테스트 (검색이 잘 되는지)
```

### 운영 중 (지속적)

```
□ 새 문서 생성 시 자동 저장
□ 문서 수정 시 자동 업데이트
□ 문서 삭제 시 자동 제거
□ 주기적 백업
□ 성능 모니터링
```

---

## 🎯 핵심 정리

```
저장 시점 (언제 저장?)
════════════════════════════════════
1. 초기 설정: 시스템 구축 시 (1회)
2. 실시간: 문서 생성/수정 시 (즉시)
3. 배치: 주기적으로 (매일/매주)
4. 사용자 업로드: 파일 업로드 시


저장 vs 검색
════════════════════════════════════
저장 (Indexing):
  - 시점: 질문 전 (사전 작업)
  - 빈도: 가끔 (문서 변경 시)
  - 시간: 느림 (분~시간)
  
검색 (Retrieval):
  - 시점: 질문 시 (실시간)
  - 빈도: 매번 (질문마다)
  - 시간: 빠름 (0.01초)


비유
════════════════════════════════════
저장 = 도서관에 책 정리
  → 한 번 정리해두면 됨
  
검색 = 도서관에서 책 찾기
  → 필요할 때마다 빠르게 찾음
```

---

**핵심: RAG 저장은 질문 "전"에 미리 해둡니다!** 🎯
