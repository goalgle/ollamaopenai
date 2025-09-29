# 🧪 테스트 가이드

ollama-agents 프로젝트의 테스트 실행 방법을 정리한 문서입니다.

---

## 📁 테스트 파일 구조

```
test/
├── README.md                        # 이 파일 (테스트 가이드)
├── rag_test_implementations.py      # 단위 테스트 (Mock 사용)
├── test_chroma_integration.py       # 통합 테스트 (ChromaDB 사용)
├── rag_performance_tests.py         # 성능 테스트
└── test_runner.py                   # 테스트 실행 스크립트
```

---

## 🎯 테스트 종류

### 1. **단위 테스트** (Unit Tests) ⚡
- **파일:** `rag_test_implementations.py`
- **VectorStore:** `MockVectorStore` (메모리 기반)
- **속도:** 매우 빠름 (< 1초)
- **의존성:** 없음 (외부 서비스 불필요)
- **마커:** `@pytest.mark.unit`

**테스트 내용:**
- ✅ 임베딩 생성 및 정규화
- ✅ 벡터 CRUD 연산 (생성, 조회, 수정, 삭제)
- ✅ 유사도 검색
- ✅ 메타데이터 필터링

### 2. **통합 테스트** (Integration Tests) 🔗
- **파일:** `test_chroma_integration.py`
- **VectorStore:** `ChromaVectorStore` (실제 ChromaDB)
- **속도:** 보통 (3-5초)
- **의존성:** ChromaDB (로컬 모드, Docker 불필요)
- **마커:** `@pytest.mark.integration`, `@pytest.mark.chroma`

**테스트 내용:**
- ✅ 실제 ChromaDB 동작 검증
- ✅ 데이터 영속성 확인
- ✅ 컬렉션 관리
- ✅ 실제 벡터 검색

### 3. **성능 테스트** (Performance Tests) 📊
- **파일:** `rag_performance_tests.py`
- **목적:** 대량 데이터 처리 성능 측정
- **마커:** `@pytest.mark.slow`

---

## 🚀 빠른 시작

### 전제 조건

```bash
# 가상환경 활성화
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

---

## 📝 테스트 실행 명령어

### ✅ 추천: 단위 테스트만 실행 (빠름!)

```bash
# 방법 1: pytest 직접 실행
pytest -m unit -v

# 방법 2: 파일 지정
pytest test/rag_test_implementations.py -v

# 방법 3: 스크립트 사용
./run_tests.sh unit
```

**예상 결과:**
```
test/rag_test_implementations.py::TestEmbeddingService::test_mock_embedding_generation PASSED
test/rag_test_implementations.py::TestEmbeddingService::test_embedding_normalization PASSED
test/rag_test_implementations.py::TestVectorStore::test_collection_management PASSED
test/rag_test_implementations.py::TestVectorStore::test_vector_operations PASSED
test/rag_test_implementations.py::TestVectorStore::test_search_filtering PASSED

==================== 8 passed in 0.45s ====================
```

---

### ✅ 통합 테스트 실행 (ChromaDB)

```bash
# ChromaDB 로컬 모드 사용 (Docker 불필요)
pytest -m integration -v

# 또는
pytest test/test_chroma_integration.py -v
```

**예상 결과:**
```
test/test_chroma_integration.py::test_chroma_connection PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_health_check PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_collection_management PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_vector_operations PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_search_filtering PASSED

==================== 5 passed in 3.21s ====================
```

---

### ✅ 전체 테스트 실행

```bash
# 모든 테스트 (단위 + 통합)
pytest -v

# 또는 스크립트로
./run_tests.sh all
```

---

### ✅ 특정 테스트만 실행

```bash
# 특정 클래스
pytest test/rag_test_implementations.py::TestVectorStore -v

# 특정 메서드
pytest test/rag_test_implementations.py::TestVectorStore::test_vector_operations -v

# 패턴 매칭
pytest -k "vector" -v          # 이름에 "vector" 포함
pytest -k "search" -v          # 이름에 "search" 포함
pytest -k "not slow" -v        # slow 제외
```

---

### ✅ 성능 테스트

```bash
# 성능 테스트 실행
pytest test/rag_performance_tests.py -v

# 느린 테스트 제외
pytest -m "not slow" -v
```

---

### ✅ 커버리지 측정

```bash
# 커버리지 리포트 생성
pytest --cov=rag --cov-report=html

# HTML 리포트 열기
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
```

### 실패 처리

```bash
# 첫 실패 시 중단
pytest -x

# 실패한 테스트만 재실행
pytest --lf

# 마지막 실패 + 새 테스트 실행
pytest --ff
```

### 성능 분석

```bash
# 느린 테스트 10개 표시
pytest --durations=10

# 병렬 실행 (pytest-xdist 필요)
pip install pytest-xdist
pytest -n auto
```

### 디버깅

```bash
# 실패 시 pdb 진입
pytest --pdb

# 로컬 변수 표시
pytest -l

# 트레이스백 길이 조절
pytest --tb=short   # 짧게
pytest --tb=long    # 길게
pytest --tb=no      # 트레이스백 없음
```

---

## 📊 테스트 마커 시스템

### 사용 가능한 마커

```python
@pytest.mark.unit         # 단위 테스트 (빠름, 의존성 없음)
@pytest.mark.integration  # 통합 테스트 (ChromaDB 필요)
@pytest.mark.chroma       # ChromaDB 사용 테스트
@pytest.mark.slow         # 느린 테스트 (성능 테스트)
```

### 마커로 필터링

```bash
# 단위 테스트만
pytest -m unit

# 통합 테스트만
pytest -m integration

# ChromaDB 테스트만
pytest -m chroma

# 통합 테스트 제외
pytest -m "not integration"

# 느린 테스트 제외
pytest -m "not slow"

# 여러 마커 조합
pytest -m "unit or integration"
```

---

## 🔍 테스트 상태 확인

### 현재 작동하는 테스트

#### ✅ 단위 테스트 (완전 동작)
```bash
pytest -m unit -v
```

**통과하는 테스트:**
- `TestEmbeddingService` - 임베딩 생성, 정규화, 유사도 계산 (5개)
- `TestVectorStore` - 컬렉션 관리, CRUD, 검색 (3개)
- `TestKnowledgeManager` - 지식 저장/조회 (일부)

#### ✅ 통합 테스트 (완전 동작)
```bash
pytest -m integration -v
```

**통과하는 테스트:**
- `test_chroma_connection` - ChromaDB 연결 확인
- `TestChromaVectorStore` - 실제 DB 동작 검증 (4개)

#### ⚠️ 알려진 이슈

**단위 테스트 일부 실패:**
```bash
# 이 2개 테스트는 Mock 설정 문제로 실패
test/rag_test_implementations.py::TestKnowledgeManager::test_agent_collection_creation
test/rag_test_implementations.py::TestKnowledgeManager::test_agent_isolation
```

**이유:** Mock 객체의 signature 불일치 (KnowledgeManager 실제 구현 필요)

---

## 🎯 상황별 추천 명령어

### 개발 중 (빠른 검증)
```bash
pytest -m unit -k "VectorStore" -v
```

### 커밋 전
```bash
pytest -m unit --cov=rag
```

### PR 생성 전
```bash
pytest --cov=rag --cov-report=html
```

### 배포 전
```bash
pytest -v --durations=10
```

---

## 🐛 문제 해결

### ❌ "No module named 'chromadb'"

```bash
pip install chromadb
```

### ❌ "No module named 'pytest'"

```bash
pip install pytest
```

### ❌ "Failed to send telemetry event"

이 경고는 무시해도 됩니다. ChromaDB의 텔레메트리 관련 경고이며 테스트 동작에는 영향 없습니다.

### ❌ "no such column: collections.topic"

```bash
# ChromaDB 버전 업그레이드
pip install --upgrade chromadb
```

### ❌ 테스트가 너무 느림

```bash
# 통합 테스트 제외하고 실행
pytest -m "not integration" -v

# 또는 단위 테스트만
pytest -m unit -v
```

---

## 📈 테스트 커버리지 현황

### 현재 커버리지 (예상)

```
rag/vector_store.py          85%  ✅
rag/knowledge_manager.py     60%  🟡
rag/utils/chunking.py        70%  🟡
```

### 커버리지 향상 목표

```bash
# 현재 커버리지 확인
pytest --cov=rag --cov-report=term-missing

# 목표: 80% 이상
```

---

## 📚 관련 문서

- **[TEST_GUIDE.md](../TEST_GUIDE.md)** - 상세한 테스트 전략
- **[TESTING_QUICKSTART.md](../TESTING_QUICKSTART.md)** - 5분 시작 가이드
- **[CHROMADB_SETUP.md](../CHROMADB_SETUP.md)** - ChromaDB 설정
- **[SUMMARY.md](../SUMMARY.md)** - 프로젝트 전체 요약

---

## 🎓 테스트 작성 예시

### 단위 테스트 작성

```python
import pytest

@pytest.mark.unit
class TestMyFeature:
    def test_with_mock(self, mock_vector_store):
        """Mock을 사용한 빠른 테스트"""
        result = mock_vector_store.create_collection("test", 384)
        assert result is True
```

### 통합 테스트 작성

```python
import pytest
import time

@pytest.mark.integration
@pytest.mark.chroma
class TestMyFeatureIntegration:
    def test_with_real_db(self, chroma_store):
        """실제 ChromaDB를 사용한 테스트"""
        collection = f"test_{int(time.time())}"
        
        try:
            result = chroma_store.create_collection(collection, 384)
            assert result is True
        finally:
            # 정리
            chroma_store.delete_collection(collection)
```

---

## 🚀 CI/CD 통합

### GitHub Actions 예시

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
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
      
      - name: Run integration tests
        run: pytest -m integration
```

---

## 📞 도움 받기

### 테스트 관련 질문

1. 먼저 이 문서를 확인하세요
2. [TEST_GUIDE.md](../TEST_GUIDE.md)에서 상세 정보 확인
3. 테스트 로그와 에러 메시지 확인

### 유용한 명령어

```bash
# pytest 도움말
pytest --help

# 사용 가능한 마커 확인
pytest --markers

# 사용 가능한 fixtures 확인
pytest --fixtures
```

---

## ✅ 체크리스트

### 개발 시작 전
- [ ] 가상환경 활성화
- [ ] `pip install -r requirements.txt` 실행
- [ ] 단위 테스트 실행 확인 (`pytest -m unit`)

### 코드 작성 후
- [ ] 관련 테스트 추가
- [ ] 단위 테스트 통과
- [ ] 커버리지 확인

### 커밋 전
- [ ] 전체 단위 테스트 통과
- [ ] 코드 포맷팅 확인
- [ ] 새 기능에 대한 테스트 추가

### PR 생성 전
- [ ] 전체 테스트 통과 (단위 + 통합)
- [ ] 커버리지 80% 이상
- [ ] 문서 업데이트

---

## 🎉 결론

### 일상적인 개발

```bash
# 이것만 기억하세요!
pytest -m unit -v
```

### 배포 전 최종 확인

```bash
# 전체 테스트 + 커버리지
pytest --cov=rag --cov-report=html
```

---

**Happy Testing! 🚀**

마지막 업데이트: 2025-09-29
