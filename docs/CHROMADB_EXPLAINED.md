# 🎨 ChromaDB 완벽 가이드

ChromaDB의 개념과 RDB와의 비교를 통해 쉽게 이해합니다.

---

## 🗄️ ChromaDB란?

**ChromaDB = 벡터 전용 데이터베이스**

```
일반 DB: 텍스트, 숫자 저장
ChromaDB: 벡터(임베딩) 저장 + 유사도 검색 특화

특징:
✅ 오픈소스 (무료!)
✅ 설치/사용 매우 쉬움
✅ Python 친화적
✅ 로컬/클라우드 모두 지원
```

---

## 📊 RDB vs ChromaDB 개념 비교

### 핵심 매핑

| RDB (PostgreSQL, MySQL) | ChromaDB | 설명 |
|------------------------|----------|------|
| **Database** | ChromaDB 인스턴스 | 최상위 컨테이너 |
| **Table** | **Collection** | 데이터 그룹 |
| **Row** | Document | 개별 데이터 항목 |
| **Column** | Embedding + Metadata | 데이터 속성 |
| **Primary Key** | ID | 고유 식별자 |
| **Index** | Vector Index (HNSW) | 검색 최적화 |
| **WHERE clause** | Metadata Filter | 조건 검색 |
| **ORDER BY similarity** | Vector Search | 유사도 정렬 |

---

## 📦 Collection (컬렉션) 개념

### RDB의 Table = ChromaDB의 Collection

```
[RDB]
Database: mydb
  ├─ Table: users
  ├─ Table: products
  └─ Table: orders

[ChromaDB]
ChromaDB Instance: ./chroma-data
  ├─ Collection: user_profiles
  ├─ Collection: product_descriptions
  └─ Collection: customer_reviews
```

### Collection의 역할

**1. 데이터 그룹화**
```python
# RDB
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    description TEXT
);

# ChromaDB
collection = chroma.create_collection(
    name="products",
    metadata={"description": "Product database"}
)
```

**2. 논리적 분리**
```
Collection: tech_docs
  - Python 문서
  - Java 문서
  - JavaScript 문서

Collection: company_policies
  - HR 정책
  - 보안 정책
  - 복지 제도

→ 목적별로 분리하여 관리!
```

**3. 검색 범위 지정**
```python
# RDB
SELECT * FROM products WHERE ...

# ChromaDB
results = collection.query(
    query_embeddings=[...],
    n_results=10
)
# → 이 컬렉션 안에서만 검색!
```

---

## 🏗️ ChromaDB 데이터 구조

### 상세 비교

```
┌─────────────────────────────────────────────────────┐
│ ChromaDB Instance (= Database)                      │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ Collection: "tech_docs" (= Table)            │  │
│  │                                              │  │
│  │  Document 1 (= Row):                        │  │
│  │    ├─ id: "doc_001"        (= Primary Key)  │  │
│  │    ├─ embedding: [0.1, 0.5, ...] (= Vector) │  │
│  │    ├─ document: "Python is..."    (= Text)  │  │
│  │    └─ metadata: {                           │  │
│  │         "category": "programming",          │  │
│  │         "language": "python",               │  │
│  │         "created": "2024-09-29"             │  │
│  │       }                                      │  │
│  │                                              │  │
│  │  Document 2:                                │  │
│  │    ├─ id: "doc_002"                         │  │
│  │    ├─ embedding: [0.2, 0.6, ...]           │  │
│  │    ├─ document: "Java is..."                │  │
│  │    └─ metadata: {...}                       │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ Collection: "customer_reviews"               │  │
│  │    ... (다른 컬렉션)                          │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Document (문서) 구조

### RDB Row vs ChromaDB Document

**RDB Table:**
```sql
CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    price DECIMAL,
    category VARCHAR(100)
);

INSERT INTO products VALUES (
    1,
    'Laptop',
    'High-performance laptop',
    1500.00,
    'electronics'
);
```

**ChromaDB Collection:**
```python
collection.add(
    ids=["prod_001"],                    # Primary Key
    embeddings=[[0.1, 0.5, 0.8, ...]],  # Vector (핵심!)
    documents=["High-performance laptop"], # Text
    metadatas=[{                          # Additional columns
        "name": "Laptop",
        "price": 1500.00,
        "category": "electronics"
    }]
)
```

### ChromaDB Document의 4가지 구성요소

```python
{
    "id": "doc_001",              # 1. 고유 ID (필수)
    
    "embedding": [0.1, 0.5, ...], # 2. 벡터 (필수) ← 핵심!
                                   #    384~1536 차원의 숫자 배열
    
    "document": "원본 텍스트",     # 3. 텍스트 (선택)
                                   #    사람이 읽을 원본 내용
    
    "metadata": {                 # 4. 메타데이터 (선택)
        "category": "tech",       #    추가 속성들
        "author": "John",
        "date": "2024-09-29"
    }
}
```

---

## 🔍 검색 비교: SQL vs ChromaDB

### 1. 일반 검색 (키워드)

**SQL:**
```sql
-- 키워드 검색
SELECT * FROM documents 
WHERE content LIKE '%Python%'
  AND category = 'programming'
ORDER BY created_date DESC
LIMIT 10;
```

**ChromaDB:**
```python
# 의미 검색
results = collection.query(
    query_embeddings=[embedding_of("Python programming")],
    where={"category": "programming"},  # 필터
    n_results=10
)
```

**차이점:**
```
SQL:
❌ "Python" 단어가 정확히 있어야 함
❌ "파이썬", "python", "PYTHON" 다 다름
❌ "프로그래밍 언어"는 못 찾음

ChromaDB:
✅ "Python", "파이썬", "프로그래밍 언어" 모두 찾음
✅ 의미가 비슷하면 찾음
✅ 동의어 자동 인식
```

### 2. 유사도 검색

**SQL: 불가능** ❌
```sql
-- 비슷한 문서 찾기?
-- → SQL로는 거의 불가능!
SELECT * FROM documents 
WHERE similar_to('Python tutorial')
-- ↑ 이런 기능 없음
```

**ChromaDB: 주 목적!** ✅
```python
# 비슷한 문서 찾기
query_embedding = embed("Python tutorial")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# 결과:
# 1. "Python 입문 가이드" (유사도 0.95)
# 2. "파이썬 기초 강좌" (유사도 0.92)
# 3. "Python for Beginners" (유사도 0.89)
# 4. "프로그래밍 시작하기" (유사도 0.85)
# 5. "코딩 튜토리얼" (유사도 0.82)
```

---

## 🎯 Collection 사용 전략

### 언제 Collection을 분리할까?

**좋은 예시 ✅**

```python
# 1. 목적별 분리
collection_faqs = chroma.create_collection("customer_faqs")
collection_manuals = chroma.create_collection("product_manuals")
collection_policies = chroma.create_collection("company_policies")

# 2. 언어별 분리
collection_ko = chroma.create_collection("docs_korean")
collection_en = chroma.create_collection("docs_english")
collection_ja = chroma.create_collection("docs_japanese")

# 3. 프로젝트별 분리
collection_project_a = chroma.create_collection("project_a_docs")
collection_project_b = chroma.create_collection("project_b_docs")

# 4. 데이터 타입별 분리
collection_images = chroma.create_collection("image_embeddings")
collection_text = chroma.create_collection("text_embeddings")
collection_audio = chroma.create_collection("audio_embeddings")
```

**나쁜 예시 ❌**

```python
# 너무 세분화
collection_python_docs_2024_09 = ...
collection_python_docs_2024_10 = ...
# → 이건 metadata로 구분!

# 혼합
collection_everything = ...  # 모든 걸 하나에
# → 검색 품질 저하, 관리 어려움
```

### 권장 전략

```python
# Collection: 큰 카테고리
# Metadata: 세부 분류

collection = chroma.create_collection("tech_docs")

collection.add(
    ids=["doc_001"],
    embeddings=[...],
    documents=["Python tutorial"],
    metadatas=[{
        "language": "python",      # 언어
        "type": "tutorial",        # 타입
        "level": "beginner",       # 난이도
        "year": 2024,              # 연도
        "month": 9                 # 월
    }]
)

# 검색 시 metadata로 필터링
results = collection.query(
    query_embeddings=[...],
    where={
        "language": "python",
        "level": "beginner"
    }
)
```

---

## 🔧 ChromaDB 주요 기능

### 1. Collection 생성

```python
from rag.vector_store import ChromaVectorStore

store = ChromaVectorStore(
    persist_directory="./chroma-data",
    use_remote=False
)

# Collection 생성
store.create_collection(
    name="my_docs",
    dimension=384,           # 벡터 차원
    metadata={"description": "My documents"}
)
```

### 2. 데이터 추가

```python
# 단건 추가
store.add_vectors(
    collection_name="my_docs",
    ids=["doc_001"],
    embeddings=[[0.1, 0.5, 0.8, ...]],
    documents=["Python is great"],
    metadatas=[{"category": "programming"}]
)

# 대량 추가
store.add_vectors(
    collection_name="my_docs",
    ids=["doc_001", "doc_002", "doc_003"],
    embeddings=[
        [0.1, 0.5, ...],
        [0.2, 0.6, ...],
        [0.3, 0.7, ...]
    ],
    documents=[
        "Python tutorial",
        "Java guide",
        "JavaScript basics"
    ],
    metadatas=[
        {"lang": "python"},
        {"lang": "java"},
        {"lang": "javascript"}
    ]
)
```

### 3. 검색

```python
# 유사도 검색
results = store.search_vectors(
    collection_name="my_docs",
    query_embedding=[0.1, 0.5, 0.8, ...],
    limit=10
)

# 필터링 + 유사도 검색
results = store.search_vectors(
    collection_name="my_docs",
    query_embedding=[0.1, 0.5, 0.8, ...],
    limit=10,
    where={"category": "programming"}  # 필터
)

# 결과
for result in results:
    print(f"ID: {result['id']}")
    print(f"유사도: {result['similarity_score']}")
    print(f"내용: {result['content']}")
    print(f"메타: {result['metadata']}")
```

### 4. 업데이트

```python
# 문서 업데이트
store.update_vectors(
    collection_name="my_docs",
    ids=["doc_001"],
    embeddings=[[0.2, 0.6, 0.9, ...]],  # 새 벡터
    documents=["Updated content"],
    metadatas=[{"updated": True}]
)
```

### 5. 삭제

```python
# 특정 문서 삭제
store.delete_vectors(
    collection_name="my_docs",
    ids=["doc_001", "doc_002"]
)

# 조건부 삭제
store.delete_vectors(
    collection_name="my_docs",
    where={"category": "deprecated"}
)
```

### 6. Collection 관리

```python
# Collection 목록 조회
collections = store.list_collections()
print(collections)  # ['my_docs', 'tech_docs', ...]

# Collection 정보
info = store.get_collection_info("my_docs")
print(f"문서 개수: {info['count']}")
print(f"차원: {info['dimension']}")

# Collection 삭제
store.delete_collection("old_collection")
```

---

## 💾 저장 방식

### 로컬 저장 (Persistent)

```python
# 데이터가 디스크에 저장됨
store = ChromaVectorStore(
    persist_directory="./chroma-data",  # 저장 경로
    use_remote=False
)

# 파일 구조:
# ./chroma-data/
#   ├─ chroma.sqlite3          # 메타데이터
#   ├─ my_docs/                # Collection별 폴더
#   │   ├─ data_level0.bin     # 벡터 데이터
#   │   └─ index/              # 인덱스
#   └─ tech_docs/
#       └─ ...
```

### 메모리 저장 (Ephemeral)

```python
# 데이터가 메모리에만 존재 (재시작 시 삭제)
import chromadb

client = chromadb.Client()  # 메모리 모드
collection = client.create_collection("temp")
```

### 원격 저장 (Server Mode)

```python
# ChromaDB 서버에 연결
store = ChromaVectorStore(
    persist_directory="./chroma-data",
    use_remote=True,
    host="localhost",
    port=8000
)
```

---

## 🎨 실전 예제

### 예제 1: 블로그 검색 시스템

```python
# 1. Collection 생성
store.create_collection("blog_posts", dimension=384)

# 2. 블로그 포스트 저장
posts = [
    {
        "id": "post_001",
        "title": "Python 튜토리얼",
        "content": "Python은 배우기 쉬운 프로그래밍 언어입니다...",
        "author": "John",
        "category": "programming",
        "date": "2024-09-29"
    },
    # ... more posts
]

for post in posts:
    embedding = embed(post['content'])
    
    store.add_vectors(
        collection_name="blog_posts",
        ids=[post['id']],
        embeddings=[embedding],
        documents=[post['content']],
        metadatas=[{
            "title": post['title'],
            "author": post['author'],
            "category": post['category'],
            "date": post['date']
        }]
    )

# 3. 검색
query = "프로그래밍 배우는 방법"
query_embedding = embed(query)

results = store.search_vectors(
    collection_name="blog_posts",
    query_embedding=query_embedding,
    limit=5,
    where={"category": "programming"}  # 프로그래밍 카테고리만
)

print("검색 결과:")
for result in results:
    print(f"제목: {result['metadata']['title']}")
    print(f"저자: {result['metadata']['author']}")
    print(f"유사도: {result['similarity_score']:.4f}")
    print()
```

### 예제 2: 다국어 문서 검색

```python
# Collection: 언어 통합
store.create_collection("multilingual_docs", dimension=384)

# 한글, 영어, 일본어 문서 저장
docs = [
    {"id": "ko_001", "text": "인공지능 기술", "lang": "ko"},
    {"id": "en_001", "text": "Artificial Intelligence", "lang": "en"},
    {"id": "ja_001", "text": "人工知能", "lang": "ja"}
]

for doc in docs:
    embedding = multilingual_embed(doc['text'])
    
    store.add_vectors(
        collection_name="multilingual_docs",
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['text']],
        metadatas=[{"language": doc['lang']}]
    )

# 한글 검색 → 영어/일본어 문서도 찾음!
query = "AI 기술"
query_embedding = multilingual_embed(query)

results = store.search_vectors(
    collection_name="multilingual_docs",
    query_embedding=query_embedding,
    limit=10
)

# 결과:
# 1. "인공지능 기술" (ko, 0.98)
# 2. "Artificial Intelligence" (en, 0.95)
# 3. "人工知能" (ja, 0.93)
```

---

## 📊 성능 특징

### ChromaDB의 장점

```
✅ 속도
- 100만 개 문서에서 0.01초 검색
- HNSW 인덱스 사용

✅ 확장성
- 수억 개 벡터 지원
- 수평 확장 가능

✅ 사용성
- 설치 간단: pip install chromadb
- API 직관적
- Python 친화적

✅ 비용
- 오픈소스 (무료!)
- 로컬 실행 가능
```

### 제한사항

```
❌ 복잡한 JOIN 불가
- RDB처럼 여러 테이블 조인 안 됨
- 해결: Collection을 하나로 통합하거나, 앱 레벨에서 처리

❌ 트랜잭션 제한적
- ACID 트랜잭션 완벽 지원 X
- 해결: 중요한 데이터는 RDB와 병행

❌ 복잡한 집계 쿼리
- GROUP BY, SUM, AVG 등 제한적
- 해결: 검색 후 Python에서 처리
```

---

## 🎯 RDB vs ChromaDB: 언제 무엇을 쓸까?

### RDB를 쓰는 경우 (PostgreSQL, MySQL)

```
✅ 정확한 데이터 관리 필요
✅ 복잡한 관계 (JOIN)
✅ 트랜잭션 중요
✅ 정형 데이터
✅ 숫자 계산, 집계

예시:
- 주문 관리
- 재고 관리
- 회계 시스템
- 고객 정보
```

### ChromaDB를 쓰는 경우

```
✅ 의미 검색 필요
✅ 비정형 데이터 (텍스트, 이미지)
✅ 유사도 기반 검색/추천
✅ AI/ML 애플리케이션
✅ RAG 시스템

예시:
- 문서 검색
- 챗봇
- 추천 시스템
- 이미지 검색
- 음악 추천
```

### 함께 쓰는 경우 (Hybrid) 🎯

```python
# PostgreSQL: 정형 데이터
# ChromaDB: 문서 검색

# 1. 사용자 정보 (PostgreSQL)
users_db.insert({
    "id": 1,
    "name": "John",
    "email": "john@example.com"
})

# 2. 사용자의 문서 (ChromaDB)
chroma.add_vectors(
    collection_name="user_documents",
    ids=["doc_001"],
    embeddings=[embedding],
    documents=["My important note"],
    metadatas={"user_id": 1}  # PostgreSQL과 연결!
)

# 3. 검색 시
# - ChromaDB에서 관련 문서 찾기
docs = chroma.search(...)

# - 각 문서의 user_id로 PostgreSQL에서 사용자 정보 가져오기
for doc in docs:
    user = users_db.get(doc['metadata']['user_id'])
    print(f"{user['name']}의 문서: {doc['content']}")
```

---

## 🎓 정리

### Collection = RDB의 Table

```
[PostgreSQL]
Database
  └─ Table (users, products, orders)

[ChromaDB]
Instance
  └─ Collection (user_profiles, product_docs, reviews)
```

### 핵심 차이

| 항목 | RDB | ChromaDB |
|------|-----|----------|
| **주 목적** | 정형 데이터 관리 | 벡터 검색 |
| **검색 방식** | 정확한 매칭 | 유사도 매칭 |
| **데이터 타입** | 텍스트, 숫자, 날짜 | 벡터 + 메타데이터 |
| **쿼리** | SQL | Python API |
| **JOIN** | 가능 | 불가능 |
| **유사도 검색** | 거의 불가능 | 주 기능 |

### 선택 가이드

```
질문: "정확한 값을 찾아야 하나?"
YES → RDB
NO  → ChromaDB

질문: "의미가 비슷한 걸 찾아야 하나?"
YES → ChromaDB
NO  → RDB

질문: "AI 기능이 필요한가?"
YES → ChromaDB
NO  → RDB

최선: 둘 다 사용! (Hybrid)
```

---

**이제 ChromaDB를 완전히 이해했습니다!** 🎉
