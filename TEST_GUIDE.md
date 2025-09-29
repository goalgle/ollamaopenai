# 테스트 실행 가이드

ollama-agents 프로젝트의 테스트 전략과 실행 방법을 설명합니다.

## 📋 테스트 구조

```
test/
├── rag_test_implementations.py    # 단위 테스트 (Mock 사용)
└── test_chroma_integration.py      # 통합 테스트 (실제 ChromaDB)
```

---

## 🎯 테스트 전략

### 1. 단위 테스트 (Unit Tests) - Mock 사용

**파일:** `test/rag_test_implementations.py`

**특징:**
- ⚡ **매우 빠름** (< 1초)
- 🧪 **메모리 기반** MockVectorStore 사용
- ❌ **외부 의존성 없음** (ChromaDB 불필요)
- 🎯 **로직 검증** 중심

**용도:**
- 개발 중 빠른 피드백
- CI/CD 파이프라인
- 로컬 개발 환경

**마커:** `@pytest.mark.unit`

---

### 2. 통합 테스트 (Integration Tests) - 실제 ChromaDB

**파일:** `test/test_chroma_integration.py`

**특징:**
- 🐢 **상대적으로 느림** (5-10초)
- 💾 **실제 ChromaDB** 서버 사용
- ✅ **외부 의존성 필요** (Docker 컨테이너)
- 🔍 **실제 동작 검증**

**용도:**
- 프로덕션 배포 전 최종 검증
- 실제 DB 호환성 확인
- 성능 테스트

**마커:** `@pytest.mark.integration`, `@pytest.mark.chroma`

---

## 🚀 테스트 실행 방법

### 전체 테스트 실행

```bash
# 모든 테스트 실행 (단위 + 통합)
pytest

# 상세 출력
pytest -v

# 특정 디렉토리만
pytest test/
```

---

### 1️⃣ 단위 테스트만 실행 (빠름)

```bash
# Mock을 사용하는 단위 테스트만 실행
pytest -m unit

# 또는 파일 직접 지정
pytest test/rag_test_implementations.py -v
```

**장점:**
- ChromaDB 없이 실행 가능
- 매우 빠른 실행 속도
- 로컬 개발에 최적

**예상 출력:**
```
test/rag_test_implementations.py::TestEmbeddingService::test_mock_embedding_generation PASSED
test/rag_test_implementations.py::TestEmbeddingService::test_embedding_normalization PASSED
test/rag_test_implementations.py::TestVectorStore::test_collection_management PASSED
test/rag_test_implementations.py::TestVectorStore::test_vector_operations PASSED

==================== 4 passed in 0.45s ====================
```

---

### 2️⃣ 통합 테스트만 실행 (ChromaDB 필요)

**⚠️ 사전 준비:** ChromaDB Docker 컨테이너 실행 필수!

```bash
# 1. ChromaDB 시작
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma

# 2. 통합 테스트 실행
pytest -m integration

# 또는 ChromaDB 테스트만
pytest -m chroma

# 또는 파일 직접 지정
pytest test/test_chroma_integration.py -v
```

**예상 출력:**
```
test/test_chroma_integration.py::test_chroma_connection PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_health_check PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_collection_management PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_vector_operations PASSED

==================== 4 passed in 3.21s ====================
```

---

### 3️⃣ 특정 테스트만 실행

```bash
# 특정 클래스
pytest test/rag_test_implementations.py::TestVectorStore -v

# 특정 메서드
pytest test/rag_test_implementations.py::TestVectorStore::test_vector_operations -v

# 패턴 매칭
pytest -k "vector" -v  # 이름에 "vector" 포함된 테스트만
pytest -k "not slow" -v  # slow 마커 제외
```

---

### 4️⃣ 통합 테스트 제외하고 실행 (빠른 실행)

```bash
# 통합 테스트 제외 (ChromaDB 없을 때)
pytest -m "not integration"

# 또는 단위 테스트만
pytest -m unit
```

---

## 🔍 ChromaDB 연결 테스트

통합 테스트 전에 ChromaDB가 정상 동작하는지 먼저 확인하세요.

```bash
# 빠른 연결 테스트
python test/test_chroma_integration.py

# 또는
pytest test/test_chroma_integration.py::test_chroma_connection -v
```

**성공 시:**
```
✅ ChromaDB 연결 성공!
   서버: localhost:8000
   상태: 정상
   기존 컬렉션 수: 2개
```

**실패 시:**
```
❌ ChromaDB 연결 실패!
   에러: Connection refused

해결 방법:
   1. Docker 컨테이너 실행 확인:
      docker ps | grep chroma
   2. 컨테이너가 없다면 실행:
      docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma
```

---

## 📊 커버리지 리포트

```bash
# 커버리지 측정
pytest --cov=rag --cov-report=html

# HTML 리포트 확인
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## 🎨 유용한 pytest 옵션

### 출력 제어

```bash
# 간결한 출력
pytest -q

# 상세한 출력
pytest -v

# 매우 상세한 출력
pytest -vv

# print 문 출력 보기
pytest -s

# 실패한 테스트만 재실행
pytest --lf

# 처음 실패 시 중단
pytest -x
```

### 성능 관련

```bash
# 느린 테스트 10개 표시
pytest --durations=10

# 병렬 실행 (pytest-xdist 필요)
pytest -n auto

# 느린 테스트 제외
pytest -m "not slow"
```

### 디버깅

```bash
# 실패 시 pdb 진입
pytest --pdb

# 특정 테스트에 브레이크포인트
pytest --trace

# 실패한 테스트의 로컬 변수 보기
pytest -l
```

---

## 🔄 CI/CD 파이프라인 권장 설정

### GitHub Actions 예시

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests
        run: pytest -m unit --cov=rag
  
  integration-tests:
    runs-on: ubuntu-latest
    services:
      chromadb:
        image: chromadb/chroma
        ports:
          - 8000:8000
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run integration tests
        run: pytest -m integration
```

---

## 🐛 문제 해결

### ❌ "No module named 'chromadb'"

```bash
pip install chromadb
```

### ❌ "Connection refused" (통합 테스트)

```bash
# ChromaDB 컨테이너 확인
docker ps | grep chroma

# 없다면 시작
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma

# 이미 있지만 중지됐다면
docker start chromadb
```

### ❌ "Port 8000 already in use"

```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# 다른 포트로 ChromaDB 실행
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8001:8000 chromadb/chroma

# 테스트 코드에서 포트 변경
# test/test_chroma_integration.py
store = ChromaVectorStore(host="localhost", port=8001, use_remote=True)
```

### ❌ "Collection not found"

```bash
# 테스트 실패 후 남은 컬렉션 정리
python -c "
from rag.vector_store import ChromaVectorStore
store = ChromaVectorStore('localhost', 8000, True)
for col in store.list_collections():
    if 'test_' in col:
        store.delete_collection(col)
"
```

---

## 📝 테스트 작성 가이드

### 단위 테스트 작성 (Mock 사용)

```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_with_mock(self, mock_vector_store):
        # Mock 사용 - 빠른 테스트
        result = mock_vector_store.create_collection("test", 384)
        assert result is True
```

### 통합 테스트 작성 (실제 ChromaDB)

```python
import pytest

@pytest.mark.integration
@pytest.mark.chroma
class TestMyFeatureIntegration:
    def test_with_real_db(self, chroma_store):
        # 실제 DB 사용 - 완전한 동작 검증
        import time
        collection = f"test_{int(time.time())}"
        
        try:
            result = chroma_store.create_collection(collection, 384)
            assert result is True
        finally:
            # 정리
            chroma_store.delete_collection(collection)
```

---

## 🎯 추천 워크플로우

### 개발 중
```bash
# 1. 빠르게 로직 검증
pytest -m unit

# 2. 특정 기능 집중 테스트
pytest -k "vector_operations" -v
```

### 커밋 전
```bash
# 전체 단위 테스트 실행
pytest -m unit --cov=rag
```

### PR/배포 전
```bash
# 1. ChromaDB 시작
docker start chromadb

# 2. 전체 테스트 (단위 + 통합)
pytest --cov=rag

# 3. 통합 테스트만 다시 확인
pytest -m integration -v
```

---

## 📚 참고 자료

- [pytest 공식 문서](https://docs.pytest.org/)
- [ChromaDB 설정 가이드](./CHROMADB_SETUP.md)
- [RAG 테스트 전략](./design/rag_testing_strategy.md)

---

**🎉 이제 테스트를 실행할 준비가 되었습니다!**

```bash
# 시작하기
pytest -m unit -v  # 빠른 단위 테스트
```
