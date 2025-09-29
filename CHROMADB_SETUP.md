# ChromaDB VectorStore 설정 가이드

ollama-agents 프로젝트에서 실제 ChromaDB를 사용하여 RAG 시스템을 구축하는 방법을 설명합니다.

## 📋 목차

1. [ChromaDB란?](#chromadb란)
2. [설치 및 실행](#설치-및-실행)
3. [사용 방법](#사용-방법)
4. [테스트 실행](#테스트-실행)
5. [문제 해결](#문제-해결)

---

## ChromaDB란?

**ChromaDB**는 AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스입니다.

### 주요 특징
- 🐍 **Python 네이티브**: pip으로 간단 설치
- 🚀 **빠른 시작**: Docker 컨테이너로 즉시 실행
- 💾 **영속성**: 데이터가 자동으로 저장됨
- 🔍 **메타데이터 필터링**: 조건부 검색 지원
- 🎯 **RAG 최적화**: 벡터 검색에 특화된 설계

---

## 설치 및 실행

### 1️⃣ ChromaDB Docker 컨테이너 실행

```bash
docker run -d \
  --name chromadb \
  -v ./chroma-data:/chroma/chroma \
  -p 8000:8000 \
  chromadb/chroma
```

### 2️⃣ 컨테이너 상태 확인

```bash
docker ps | grep chroma
curl http://localhost:8000/api/v1/heartbeat
```

### 3️⃣ Python 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 사용 방법

### 기본 사용 예제

```python
from rag.vector_store import ChromaVectorStore

# ChromaDB 연결
vector_store = ChromaVectorStore(
    host="localhost",
    port=8000,
    use_remote=True
)

# 컬렉션 생성
vector_store.create_collection("my_agent", dimension=384)

# 문서 추가
vector_store.add_vectors(
    collection_name="my_agent",
    ids=["doc1"],
    embeddings=[[0.1, 0.2, ...]],
    metadatas=[{"topic": "python"}],
    documents=["Python is great"]
)

# 검색
results = vector_store.search_vectors(
    collection_name="my_agent",
    query_embedding=[0.1, 0.2, ...],
    limit=5
)
```

### 실제 예제 실행

```bash
python examples/chroma_example.py
```

---

## 테스트 실행

### 연결 테스트

```bash
python test/test_chroma_integration.py
```

### 전체 통합 테스트

```bash
pytest test/test_chroma_integration.py -v
```

---

## 문제 해결

### ❌ 연결 실패

```bash
# 컨테이너 시작
docker start chromadb
```

### ❌ 포트 충돌

```bash
# 다른 포트 사용
docker run -d -v ./chroma-data:/chroma/chroma -p 8001:8000 chromadb/chroma
```

---

## 다음 단계

1. ✅ 임베딩 서비스 구현
2. ✅ KnowledgeManager 통합
3. ✅ 에이전트 구축

**참고:** [ChromaDB 공식 문서](https://docs.trychroma.com/)
