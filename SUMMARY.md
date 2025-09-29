# 📊 프로젝트 구조 및 테스트 전략 요약

Mock VectorStore와 실제 ChromaDB를 분리하여 효율적인 테스트 환경을 구축했습니다.

---

## 🎯 핵심 개념

### 1. **Mock VectorStore** (메모리 DB)
- **위치:** `test/rag_test_implementations.py`
- **역할:** 단위 테스트용 가짜 DB
- **저장소:** Python 딕셔너리 (메모리)
- **특징:**
  - ⚡ 매우 빠름 (< 1초)
  - ❌ 외부 의존성 없음
  - 🧪 로직 검증 중심
  - 💾 테스트 종료 시 데이터 사라짐

### 2. **ChromaVectorStore** (실제 DB)
- **위치:** `rag/vector_store.py`
- **역할:** 프로덕션용 실제 벡터 DB
- **저장소:** ChromaDB (디스크)
- **특징:**
  - 💾 데이터 영속성
  - ✅ 프로덕션 환경 사용
  - 🔍 실제 DB 호환성 검증
  - 🐢 상대적으로 느림 (네트워크 I/O)

---

## 📁 파일 구조

```
ollama-agents/
├── rag/
│   ├── __init__.py
│   ├── vector_store.py              # ✅ 실제 ChromaDB 구현
│   └── knowledge_manager.py
│
├── test/
│   ├── rag_test_implementations.py  # 🧪 단위 테스트 (Mock)
│   └── test_chroma_integration.py   # ✅ 통합 테스트 (ChromaDB)
│
├── examples/
│   └── chroma_example.py            # 📚 사용 예제
│
├── pytest.ini                        # ⚙️ pytest 설정
├── run_tests.sh                      # 🚀 테스트 실행 스크립트
├── TEST_GUIDE.md                     # 📖 상세 테스트 가이드
├── TESTING_QUICKSTART.md            # ⚡ 빠른 시작 가이드
├── CHROMADB_SETUP.md                # 🐳 ChromaDB 설정 가이드
└── requirements.txt                 # 📦 의존성 (chromadb 추가됨)
```

---

## 🔄 테스트 워크플로우

### 개발 단계별 테스트 전략

```
개발 시작
   ↓
[1] 단위 테스트 (Mock)
   pytest -m unit
   ⚡ 빠른 피드백 (< 1초)
   ❌ ChromaDB 불필요
   ↓
[2] 기능 개발
   ↓
[3] 통합 테스트 (ChromaDB)
   pytest -m integration
   ✅ 실제 DB 검증 (5-10초)
   🐳 Docker 필요
   ↓
[4] PR/배포
   pytest --cov=rag
   📊 전체 테스트 + 커버리지
```

---

## 🚀 빠른 실행 가이드

### 1️⃣ 단위 테스트 (즉시 실행 가능)

```bash
# Mock을 사용한 빠른 테스트
pytest -m unit -v

# 또는 스크립트로
chmod +x run_tests.sh
./run_tests.sh unit
```

**✅ 장점:**
- ChromaDB 없이 실행
- 매우 빠른 속도
- CI/CD에 최적

**📝 테스트 내용:**
- 임베딩 생성/정규화
- 벡터 CRUD 연산
- 검색 및 필터링 로직

---

### 2️⃣ 통합 테스트 (ChromaDB 필요)

```bash
# 1단계: ChromaDB 시작
docker run -d \
  --name chromadb \
  -v ./chroma-data:/chroma/chroma \
  -p 8000:8000 \
  chromadb/chroma

# 2단계: 연결 확인
python test/test_chroma_integration.py

# 3단계: 통합 테스트 실행
pytest -m integration -v

# 또는 스크립트로
./run_tests.sh integration
```

**✅ 장점:**
- 실제 DB 동작 검증
- 성능 측정 가능
- 프로덕션 배포 전 최종 확인

**📝 테스트 내용:**
- ChromaDB 연결
- 실제 벡터 저장/검색
- 메타데이터 필터링
- 다중 컬렉션 격리

---

## 📊 테스트 비교표

| 항목 | 단위 테스트 (Mock) | 통합 테스트 (ChromaDB) |
|------|-------------------|----------------------|
| **파일** | `rag_test_implementations.py` | `test_chroma_integration.py` |
| **VectorStore** | `MockVectorStore` | `ChromaVectorStore` |
| **마커** | `@pytest.mark.unit` | `@pytest.mark.integration` |
| **속도** | ⚡ < 1초 | 🐢 5-10초 |
| **의존성** | ❌ 없음 | ✅ Docker + ChromaDB |
| **저장소** | 메모리 (딕셔너리) | 디스크 (ChromaDB) |
| **영속성** | ❌ 테스트 종료 시 삭제 | ✅ 데이터 유지 |
| **용도** | 로직 검증, 빠른 개발 | 실제 동작 검증, 최종 확인 |
| **CI/CD** | ✅ 적합 | ⚠️ Docker 환경 필요 |

---

## 🎨 pytest 마커 시스템

### 마커 정의 (`pytest.ini`)

```ini
markers =
    unit: 단위 테스트 (Mock 사용, 외부 의존성 없음)
    integration: 통합 테스트 (실제 ChromaDB 필요)
    slow: 느린 테스트 (성능 테스트, 대량 데이터)
    chroma: ChromaDB가 필요한 테스트
```

### 마커 사용 예시

```python
# 단위 테스트
@pytest.mark.unit
class TestVectorStore:
    def test_collection_management(self, mock_vector_store):
        ...

# 통합 테스트
@pytest.mark.integration
@pytest.mark.chroma
class TestChromaVectorStore:
    def test_collection_management(self, chroma_store):
        ...
```

### 마커로 테스트 필터링

```bash
# 단위 테스트만
pytest -m unit

# 통합 테스트만
pytest -m integration

# ChromaDB 테스트만
pytest -m chroma

# 통합 테스트 제외
pytest -m "not integration"

# 여러 마커 조합
pytest -m "unit or integration"
```

---

## 🔧 주요 명령어 모음

### 개발 중

```bash
# 빠른 검증
pytest -m unit

# 특정 테스트
pytest -k "vector_operations" -v

# 실패 시 즉시 중단
pytest -x
```

### 커밋 전

```bash
# 전체 단위 테스트
pytest -m unit --cov=rag

# 또는
./run_tests.sh fast
```

### PR/배포 전

```bash
# 전체 테스트
./run_tests.sh all

# 커버리지 리포트
./run_tests.sh coverage
open htmlcov/index.html
```

### ChromaDB 관리

```bash
# 시작
docker start chromadb

# 중지
docker stop chromadb

# 재시작
docker restart chromadb

# 로그 확인
docker logs chromadb

# 삭제 (데이터 유지)
docker rm chromadb

# 완전 삭제 (데이터 포함)
docker rm -f chromadb
rm -rf chroma-data/
```

---

## 🎯 실제 사용 예시

### Mock 사용 (단위 테스트)

```python
from test.rag_test_implementations import MockVectorStore

# 메모리 기반 테스트
mock_store = MockVectorStore()
mock_store.create_collection("test", 768)
mock_store.add_vectors("test", ["id1"], [[0.1, 0.2, ...]], ...)

# 빠르고 격리된 테스트
```

### ChromaDB 사용 (통합 테스트)

```python
from rag.vector_store import ChromaVectorStore

# 실제 DB 연결
store = ChromaVectorStore(
    host="localhost",
    port=8000,
    use_remote=True
)

# 실제 동작 검증
store.create_collection("prod_collection", 384)
store.add_vectors("prod_collection", ids, embeddings, ...)
results = store.search_vectors("prod_collection", query, limit=5)
```

### 프로덕션 코드

```python
from rag.vector_store import ChromaVectorStore

# 실제 RAG 시스템에서 사용
vector_store = ChromaVectorStore(
    host="chromadb.production.com",
    port=8000,
    use_remote=True
)

# 에이전트별 컬렉션
vector_store.create_collection(f"agent-{agent_id}", 384)

# 지식 저장
vector_store.add_vectors(
    collection_name=f"agent-{agent_id}",
    ids=document_ids,
    embeddings=document_embeddings,
    metadatas=document_metadatas,
    documents=document_texts
)

# 검색
results = vector_store.search_vectors(
    collection_name=f"agent-{agent_id}",
    query_embedding=question_embedding,
    limit=5,
    where={"category": "python"}  # 필터링
)
```

---

## 📚 문서 가이드

### 빠르게 시작하기
👉 **[TESTING_QUICKSTART.md](./TESTING_QUICKSTART.md)**
- 5분 안에 테스트 시작
- 단계별 명령어
- 문제 해결 팁

### 상세한 테스트 가이드
👉 **[TEST_GUIDE.md](./TEST_GUIDE.md)**
- 테스트 전략 설명
- pytest 옵션 전체
- CI/CD 설정 예시
- 테스트 작성 가이드

### ChromaDB 설정
👉 **[CHROMADB_SETUP.md](./CHROMADB_SETUP.md)**
- Docker 설치 및 실행
- 연결 테스트
- 문제 해결
- 데이터 관리

### 사용 예제
👉 **[examples/chroma_example.py](./examples/chroma_example.py)**
- Python 지식베이스 구축
- 질문-검색 워크플로우
- 메타데이터 필터링 예시

---

## 🎓 학습 경로

### 1단계: Mock 이해하기
```bash
# Mock 구현 살펴보기
cat test/rag_test_implementations.py | grep "class MockVectorStore" -A 50

# Mock 테스트 실행
pytest test/rag_test_implementations.py::TestVectorStore -v
```

### 2단계: ChromaDB 설치
```bash
# Docker로 ChromaDB 실행
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma

# 연결 확인
curl http://localhost:8000/api/v1/heartbeat
```

### 3단계: 실제 구현 이해하기
```bash
# ChromaDB 구현 살펴보기
cat rag/vector_store.py | grep "class ChromaVectorStore" -A 50

# 통합 테스트 실행
pytest test/test_chroma_integration.py -v
```

### 4단계: 예제 실행
```bash
# 실제 동작 확인
python examples/chroma_example.py
```

### 5단계: 직접 개발
```python
# 나만의 RAG 에이전트 만들기
from rag.vector_store import ChromaVectorStore

store = ChromaVectorStore("localhost", 8000, True)
# ... 에이전트 개발
```

---

## 🔍 핵심 차이점 정리

### Mock vs ChromaDB 인터페이스

**동일한 인터페이스:**
```python
# 둘 다 동일한 메서드 제공
store.create_collection(name, dimension)
store.add_vectors(collection, ids, embeddings, metadatas, documents)
store.search_vectors(collection, query_embedding, limit, where)
store.delete_collection(name)
```

**다른 점:**
- **Mock:** 메모리에만 저장, 빠름, 테스트 후 삭제
- **ChromaDB:** 디스크에 저장, 느림, 데이터 영속성

**선택 기준:**
- **개발 중:** Mock (빠른 피드백)
- **배포 전:** ChromaDB (실제 동작 검증)
- **프로덕션:** ChromaDB (데이터 보존)

---

## ✅ 체크리스트

### 로컬 개발 환경 설정
- [ ] Python 가상환경 활성화
- [ ] `pip install -r requirements.txt` 실행
- [ ] 단위 테스트 실행 (`pytest -m unit`)
- [ ] Docker 설치 확인
- [ ] ChromaDB 컨테이너 실행
- [ ] 통합 테스트 실행 (`pytest -m integration`)

### 코드 커밋 전
- [ ] 단위 테스트 통과
- [ ] 새 테스트 추가 (새 기능이 있다면)
- [ ] 코드 포맷팅 확인
- [ ] 커버리지 확인 (> 80%)

### PR 생성 전
- [ ] 전체 테스트 통과 (`./run_tests.sh all`)
- [ ] 통합 테스트 통과
- [ ] 문서 업데이트
- [ ] 예제 코드 동작 확인

---

## 🎉 완료!

이제 Mock과 ChromaDB를 효과적으로 사용할 수 있습니다!

### 다음 단계

1. **실제 임베딩 모델 연결**
   - OpenAI Embeddings
   - HuggingFace Sentence Transformers
   
2. **KnowledgeManager 통합**
   - ChromaVectorStore 사용
   - 에이전트별 지식 관리

3. **RAG 에이전트 구축**
   - 문서 학습
   - 질문-답변 시스템
   - 컨텍스트 기반 대화

### 도움이 필요하면

- 문서를 참고하세요: `TEST_GUIDE.md`, `CHROMADB_SETUP.md`
- 예제를 실행하세요: `python examples/chroma_example.py`
- 테스트를 확인하세요: `pytest -v`

---

**Happy Testing! 🚀**
