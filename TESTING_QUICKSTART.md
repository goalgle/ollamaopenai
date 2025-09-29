# 🚀 테스트 빠른 시작 가이드

5분 안에 테스트를 시작하는 방법!

---

## ⚡ 1단계: 빠른 단위 테스트 (Mock)

**외부 의존성 없이 즉시 실행 가능!**

```bash
# 방법 1: pytest 직접 실행
pytest -m unit -v

# 방법 2: 스크립트 사용
chmod +x run_tests.sh
./run_tests.sh unit
```

**예상 시간:** < 1초  
**필요 사항:** 없음 (Mock 사용)

---

## 🐳 2단계: ChromaDB 시작

통합 테스트를 위해 ChromaDB Docker 컨테이너를 실행합니다.

```bash
# ChromaDB 시작
docker run -d \
  --name chromadb \
  -v ./chroma-data:/chroma/chroma \
  -p 8000:8000 \
  chromadb/chroma

# 상태 확인
docker ps | grep chroma
```

**또는 이미 실행 중이라면:**
```bash
docker start chromadb
```

---

## ✅ 3단계: 연결 테스트

ChromaDB가 정상 작동하는지 확인합니다.

```bash
# 연결 테스트
python test/test_chroma_integration.py
```

**성공 출력:**
```
✅ ChromaDB 연결 성공!
   서버: localhost:8000
   상태: 정상
   기존 컬렉션 수: 0개
```

---

## 🧪 4단계: 통합 테스트 실행

실제 ChromaDB를 사용하는 테스트를 실행합니다.

```bash
# 방법 1: pytest 직접 실행
pytest -m integration -v

# 방법 2: 스크립트 사용
./run_tests.sh integration

# 방법 3: ChromaDB 테스트만
pytest -m chroma -v
```

**예상 시간:** 5-10초  
**필요 사항:** ChromaDB 실행 중

---

## 📊 5단계: 전체 테스트 + 커버리지

```bash
# 전체 테스트 실행
./run_tests.sh all

# 커버리지 리포트
./run_tests.sh coverage

# HTML 리포트 보기
open htmlcov/index.html  # Mac
```

---

## 🎯 상황별 명령어

### 개발 중 (빠른 피드백)
```bash
# 단위 테스트만 (가장 빠름)
pytest -m unit
```

### 특정 기능 테스트
```bash
# Vector Store 테스트만
pytest -k "VectorStore" -v

# 검색 관련 테스트만
pytest -k "search" -v
```

### PR 전 (전체 검증)
```bash
# 모든 테스트 실행
./run_tests.sh all
```

### ChromaDB 없이 테스트
```bash
# 통합 테스트 제외
pytest -m "not integration"

# 또는
./run_tests.sh fast
```

---

## 🐛 문제 해결

### ChromaDB 연결 실패

```bash
# 컨테이너 상태 확인
docker ps -a | grep chroma

# 로그 확인
docker logs chromadb

# 재시작
docker restart chromadb

# 완전히 새로 시작
docker rm -f chromadb
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma
```

### 포트 충돌 (8000번 포트 사용 중)

```bash
# 다른 포트로 실행
docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8001:8000 chromadb/chroma

# 테스트 코드 수정 필요:
# test/test_chroma_integration.py 에서 port=8001로 변경
```

### 패키지 오류

```bash
# 가상환경 활성화 확인
source .venv/bin/activate  # Linux/Mac

# 패키지 재설치
pip install -r requirements.txt
```

---

## 📝 테스트 결과 예시

### ✅ 성공 (단위 테스트)
```
test/rag_test_implementations.py::TestVectorStore::test_collection_management PASSED
test/rag_test_implementations.py::TestVectorStore::test_vector_operations PASSED
test/rag_test_implementations.py::TestVectorStore::test_search_filtering PASSED

==================== 3 passed in 0.23s ====================
```

### ✅ 성공 (통합 테스트)
```
test/test_chroma_integration.py::test_chroma_connection PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_health_check PASSED
test/test_chroma_integration.py::TestChromaVectorStore::test_vector_operations PASSED

==================== 3 passed in 4.15s ====================
```

---

## 🎓 더 알아보기

자세한 내용은 다음 문서를 참고하세요:

- **[TEST_GUIDE.md](./TEST_GUIDE.md)** - 상세한 테스트 가이드
- **[CHROMADB_SETUP.md](./CHROMADB_SETUP.md)** - ChromaDB 설정
- **[pytest.ini](./pytest.ini)** - pytest 설정

---

## 🎉 완료!

이제 테스트를 실행할 준비가 되었습니다!

```bash
# 빠른 시작
./run_tests.sh unit
```

문제가 있으면 [문제 해결](#-문제-해결) 섹션을 확인하세요.
