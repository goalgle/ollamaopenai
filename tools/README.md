# 🛠️ ChromaDB 조회 도구

ChromaDB에 저장된 데이터를 SQL처럼 조회하고 관리하는 도구입니다.

---

## 📋 기능

- ✅ 모든 컬렉션 목록 조회
- ✅ 컬렉션 상세 정보 확인
- ✅ 모든 문서 조회
- ✅ 텍스트 검색 (유사도 기반)
- ✅ 메타데이터 필터링 검색
- ✅ 컬렉션 삭제

---

## 🚀 사용 방법

### 1️⃣ 모든 컬렉션 목록 보기

```bash
python tools/chroma_query.py --list
```

**출력 예시:**
```
📁 ChromaDB 연결: ./chroma-data

======================================================================
📚 저장된 컬렉션 목록
======================================================================
1. demo_python_agent
   문서 수: 5개

2. test_collection
   문서 수: 10개
```

---

### 2️⃣ 컬렉션 정보 확인

```bash
python tools/chroma_query.py --collection demo_python_agent --info
```

**출력 예시:**
```
======================================================================
📊 컬렉션 정보: demo_python_agent
======================================================================
총 문서 수: 5개

📄 샘플 문서 (최대 3개):
----------------------------------------------------------------------

[1] ID: py_001
    내용: Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다.
    메타: {'topic': 'history', 'difficulty': 'beginner'}

[2] ID: py_002
    내용: Python은 들여쓰기로 코드 블록을 구분하는 독특한 문법을 사용합니다.
    메타: {'topic': 'syntax', 'difficulty': 'beginner'}
```

---

### 3️⃣ 모든 문서 조회

```bash
# 모든 문서
python tools/chroma_query.py --collection demo_python_agent --show-all

# 개수 제한 (처음 3개만)
python tools/chroma_query.py --collection demo_python_agent --show-all --limit 3
```

**출력 예시:**
```
======================================================================
📄 컬렉션의 모든 문서: demo_python_agent
======================================================================
총 문서 수: 5개

[1] ID: py_001
    내용: Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다.
    메타: {'topic': 'history', 'difficulty': 'beginner'}

[2] ID: py_002
    내용: Python은 들여쓰기로 코드 블록을 구분하는 독특한 문법을 사용합니다.
    메타: {'topic': 'syntax', 'difficulty': 'beginner'}
...
```

---

### 4️⃣ 텍스트 검색 (유사도 기반)

```bash
# 기본 검색 (상위 5개)
python tools/chroma_query.py --collection demo_python_agent --search "Python의 역사"

# 개수 제한
python tools/chroma_query.py --collection demo_python_agent --search "비동기 프로그래밍" --limit 3
```

**출력 예시:**
```
======================================================================
🔍 검색: Python의 역사
   컬렉션: demo_python_agent
======================================================================

📚 검색 결과 (3개):
----------------------------------------------------------------------

[1] 유사도: 0.8756
    ID: py_001
    내용: Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다.
    메타: {'topic': 'history', 'difficulty': 'beginner'}

[2] 유사도: 0.6234
    ID: py_003
    내용: 리스트 컴프리헨션은 Python의 강력한 기능으로...
    메타: {'topic': 'advanced', 'difficulty': 'intermediate'}
```

---

### 5️⃣ 메타데이터 필터 검색

```bash
# 특정 토픽만
python tools/chroma_query.py --collection demo_python_agent --filter '{"topic": "history"}'

# 난이도 필터
python tools/chroma_query.py --collection demo_python_agent --filter '{"difficulty": "beginner"}'

# 여러 조건
python tools/chroma_query.py --collection demo_python_agent --filter '{"topic": "advanced", "difficulty": "intermediate"}'
```

**출력 예시:**
```
======================================================================
🔍 필터 검색
   컬렉션: demo_python_agent
   조건: {'topic': 'history'}
======================================================================

📚 검색 결과 (1개):
----------------------------------------------------------------------

[1] ID: py_001
    내용: Python은 1991년 귀도 반 로섬이 개발한 고수준 프로그래밍 언어입니다.
    메타: {'topic': 'history', 'difficulty': 'beginner'}
```

---

### 6️⃣ 컬렉션 삭제

```bash
python tools/chroma_query.py --collection test_collection --delete
```

**확인 프롬프트:**
```
⚠️  정말로 'test_collection' 컬렉션을 삭제하시겠습니까? (yes/no): yes
✅ 컬렉션 'test_collection'이 삭제되었습니다.
```

---

## 📊 SQL과 비교

| SQL | ChromaDB 도구 |
|-----|--------------|
| `SHOW TABLES;` | `python tools/chroma_query.py --list` |
| `SELECT * FROM table;` | `python tools/chroma_query.py --collection table --show-all` |
| `SELECT * FROM table LIMIT 5;` | `python tools/chroma_query.py --collection table --show-all --limit 5` |
| `SELECT * WHERE topic='history';` | `python tools/chroma_query.py --collection table --filter '{"topic": "history"}'` |
| `SELECT * WHERE text LIKE '%Python%';` | `python tools/chroma_query.py --collection table --search "Python"` |
| `DROP TABLE table;` | `python tools/chroma_query.py --collection table --delete` |

---

## 🔧 고급 사용

### Python 스크립트에서 사용

```python
from tools.chroma_query import ChromaQueryTool

# 도구 초기화
tool = ChromaQueryTool(persist_directory="./chroma-data")

# 컬렉션 목록
tool.list_collections()

# 컬렉션 정보
tool.collection_info("demo_python_agent")

# 모든 문서
tool.show_all_documents("demo_python_agent", limit=10)

# 검색
tool.search("demo_python_agent", "Python 역사", limit=5)

# 필터 검색
tool.filter_search("demo_python_agent", {"topic": "history"}, limit=10)

# 삭제
tool.delete_collection("test_collection", confirm=True)
```

---

## 💡 팁

### 1. 데이터 탐색

```bash
# 1단계: 어떤 컬렉션이 있는지 확인
python tools/chroma_query.py --list

# 2단계: 관심있는 컬렉션 정보 확인
python tools/chroma_query.py --collection demo_python_agent --info

# 3단계: 샘플 확인 (처음 5개)
python tools/chroma_query.py --collection demo_python_agent --show-all --limit 5
```

### 2. 검색 결과 검증

```bash
# 질문으로 검색해보고 관련 문서가 잘 나오는지 확인
python tools/chroma_query.py --collection demo_python_agent --search "리스트 정렬"
```

### 3. 메타데이터 활용

```bash
# 특정 난이도 문서만 추출
python tools/chroma_query.py --collection demo_python_agent --filter '{"difficulty": "beginner"}'
```

---

## 🐛 문제 해결

### ❌ "No module named 'rag'"

```bash
# 프로젝트 루트에서 실행하세요
cd /path/to/ollama-agents
python tools/chroma_query.py --list
```

### ❌ "Collection not found"

```bash
# 컬렉션 이름 확인
python tools/chroma_query.py --list
```

### ❌ "Failed to send telemetry event"

이 경고는 무시해도 됩니다. ChromaDB의 텔레메트리 관련 경고입니다.

---

## 📝 다른 데이터 디렉토리 사용

```bash
# 다른 위치의 ChromaDB 데이터 조회
python tools/chroma_query.py --dir /path/to/other/chroma-data --list
```

---

## 🎯 실전 예제

### 예제 1: 저장된 에이전트 지식 확인

```bash
# 1. 에이전트 컬렉션 찾기
python tools/chroma_query.py --list

# 2. agent-001 컬렉션 확인
python tools/chroma_query.py --collection agent-001 --info

# 3. 특정 주제로 검색
python tools/chroma_query.py --collection agent-001 --search "머신러닝"
```

### 예제 2: 데이터 품질 검사

```bash
# 중복 문서 찾기 (같은 내용이 여러 번 저장되었는지)
python tools/chroma_query.py --collection demo_python_agent --show-all

# 메타데이터 일관성 확인
python tools/chroma_query.py --collection demo_python_agent --filter '{"topic": "unknown"}'
```

### 예제 3: 테스트 데이터 정리

```bash
# test로 시작하는 컬렉션들 삭제
python tools/chroma_query.py --collection test_collection_1 --delete
python tools/chroma_query.py --collection test_collection_2 --delete
```

---

## 📚 관련 문서

- **[CHROMADB_SETUP.md](../CHROMADB_SETUP.md)** - ChromaDB 설치 및 설정
- **[test/README.md](../test/README.md)** - 테스트 가이드

---

**Happy Querying! 🔍**
