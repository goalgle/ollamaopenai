# ChromaDB Utility CLI - 사용 매뉴얼

## 📋 개요

`test_chroma_util.py`는 ChromaDB를 대화형으로 탐색하고 관리할 수 있는 CLI(Command Line Interface) 도구입니다.

### 주요 기능
- ✅ 콜렉션 조회 및 관리
- 🔍 유사도 기반 문서 검색
- 🎯 AND 조건 필터링 (누적 필터)
- 📝 문서 추가/삭제
- 💾 명령어 히스토리 및 자동완성
- 🌐 로컬/원격 ChromaDB 지원

---

## 🚀 실행 방법

### 기본 실행
```bash
# 기본 디렉토리 (./chroma-data)
python test_chroma_util.py

# 다른 디렉토리 지정 (위치 인자)
python test_chroma_util.py ./stock-rag-data

# 다른 디렉토리 지정 (옵션)
python test_chroma_util.py --dir ./my-chroma-data

# 원격 ChromaDB 서버 연결
python test_chroma_util.py --remote --host localhost --port 8000
python test_chroma_util.py --remote --host db.example.com --port 8000
```

### 실행 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `directory` | ChromaDB 디렉토리 (위치 인자) | `./chroma-data` |
| `--dir` | ChromaDB 디렉토리 (옵션, 위치 인자보다 우선) | - |
| `--remote` | 원격 서버 모드 사용 | `False` |
| `--host` | 원격 서버 호스트 | `localhost` |
| `--port` | 원격 서버 포트 | `8000` |

---

## 📚 명령어 레퍼런스

### 파일 임포트 명령어

#### `import <collection_name> <file_path> [options]`
파일에서 여러 문서를 한번에 임포트합니다.

```bash
chroma> import my_docs ./sample_documents.py
chroma> import my_docs ./data/documents.py --no-auto-id
chroma> import my_docs ./data/docs.py --batch-size 50
```

**매개변수:**
- `collection_name`: 콜렉션 이름 (필수)
- `file_path`: 문서 파일 경로 (필수)

**옵션:**
- `--no-auto-id`: ID 자동 생성 비활성화 (파일의 모든 문서에 id 필드 필수)
- `--batch-size <size>`: 배치 크기 지정 (기본값: 100)

**파일 포맷:**
```python
documents = [
  {
    "id": "doc_001",        # Optional: 생략 시 자동 생성
    "document": "내용...",   # Required
    "metadata": {           # Optional
      "type": "tutorial",
      "category": "python"
    }
  },
  ...
]
```

**출력 예시:**
```
📂 Loading documents from: ./sample_documents.py
✅ Loaded 8 documents from file
✅ Prepared 8 documents for import
   - Auto-generated IDs: 2
   - Custom IDs: 6

📥 Importing documents (batch_size=100)...
   Batch 1: 8 documents ✓

============================================================
✅ Import completed
   Total: 8
   Imported: 8
============================================================

💡 View imported documents:
   show my_docs 0 10
   search my_docs "your query" 10
```

---

#### `preview <file_path> [max_docs]`
파일 내용을 미리보기로 확인합니다.

```bash
chroma> preview ./sample_documents.py
chroma> preview ./data/documents.py 10
```

**매개변수:**
- `file_path`: 파일 경로 (필수)
- `max_docs`: 최대 표시 문서 개수 (기본값: 5)

**출력 예시:**
```
📄 File Preview: ./sample_documents.py
============================================================
Total documents: 8

Document 1:
  ID: collect_entity_lton_place
  Content: 
public class LtonPlace {
    private Long placeId;
    private String placeName;
  ...
  Metadata: {'type': 'entity', 'layer': 'domain', 'language': 'java'}

Document 2:
  ID: collect_service_place_service
  Content: 
@Service
public class PlaceService {
...

... and 6 more documents
============================================================

💡 To import this file:
   import <collection_name> ./sample_documents.py
```

---

### 조회 명령어

#### `collections`
모든 콜렉션 목록을 출력합니다.

```bash
chroma> collections
```

**출력 예시:**
```
📚 Collections:
  1. stock_knowledge (1234 documents)
  2. tech_docs (567 documents)
  3. my_collection (89 documents)
```

---

#### `info <collection_name>`
특정 콜렉션의 상세 정보를 출력합니다.

```bash
chroma> info stock_knowledge
```

**출력 예시:**
```
📖 Collection: stock_knowledge
   Documents: 1234
   Metadata: {...}
```

**TAB 자동완성**: 콜렉션 이름을 TAB으로 자동완성할 수 있습니다.

---

#### `show <collection_name> [start] [size]`
콜렉션의 문서들을 출력합니다.

```bash
chroma> show stock_knowledge
chroma> show stock_knowledge 0 10    # 0번부터 10개
chroma> show stock_knowledge 20 5    # 20번부터 5개
```

**매개변수:**
- `collection_name`: 콜렉션 이름 (필수)
- `start`: 시작 인덱스 (기본값: 0)
- `size`: 가져올 문서 개수 (기본값: 10)

---

#### `search <collection_name> <query> [limit]`
유사도 기반으로 문서를 검색합니다.

```bash
chroma> search stock_knowledge "Tesla stock analysis"
chroma> search stock_knowledge "Python programming" 20
chroma> search tech_docs "machine learning tutorial" 50
```

**매개변수:**
- `collection_name`: 콜렉션 이름 (필수)
- `query`: 검색 쿼리 (필수, 여러 단어 가능)
- `limit`: 최대 결과 개수 (기본값: 10)

**결과:**
- 검색 결과가 `last_results`와 `original_results`에 저장됩니다
- 이후 필터링 명령어를 사용할 수 있습니다

---

### 필터링 명령어 (AND 조건)

> ⚠️ **중요**: 모든 필터는 **AND 조건**으로 누적됩니다!

#### `filter <min_similarity>`
현재 결과를 유사도로 필터링합니다.

```bash
chroma> search stock_knowledge "Tesla" 100
# 100개 결과

chroma> filter 0.5
# 유사도 >= 0.5인 문서만 남음

chroma> filter 0.8
# 위 결과 중 유사도 >= 0.8인 문서만
```

**매개변수:**
- `min_similarity`: 최소 유사도 (float)
  - 예: `0.5` (유사도 0.5 이상)
  - 예: `0` (모든 문서)
  - 예: `-0.5` (유사도 -0.5 이상)

**출력:**
```
🔎 Filtering current results (similarity >= 0.5)
Before: 100 documents
Similarity range: -0.1234 ~ 0.9876
After filter: 45 documents
💡 Use 'reset' to go back to original search results
```

---

#### `metadata <key> <value>`
현재 결과를 메타데이터로 필터링합니다.

```bash
chroma> metadata category tech
chroma> metadata author John
chroma> metadata year 2024
```

**매개변수:**
- `key`: 메타데이터 키 (필수)
- `value`: 메타데이터 값 (필수)

**예시 시나리오:**
```bash
chroma> search tech_docs "Python tutorial" 50
chroma> filter 0.6
chroma> metadata category programming
chroma> metadata difficulty beginner
# 최종: 유사도 >= 0.6 AND category=programming AND difficulty=beginner
```

---

#### `top <count>`
현재 결과에서 유사도가 높은 순서로 상위 N개를 표시합니다.

```bash
chroma> top 5     # 상위 5개
chroma> top 10    # 상위 10개
chroma> top 3     # 상위 3개
```

**매개변수:**
- `count`: 표시할 문서 개수 (필수, 양수)

**출력:**
```
🏆 Top 3 documents by similarity
Before: 45 documents
After top: 3 documents
💡 Use 'reset' to go back to original search results

🥇 Rank 1
   ID: doc_123
   Similarity: 0.9876
   Metadata: {'category': 'tech', 'author': 'John'}
   Content: This is a Python tutorial about...

🥇 Rank 2
   ID: doc_456
   Similarity: 0.9543
   ...
```

---

#### `reset`
모든 필터를 제거하고 원본 검색 결과로 돌아갑니다.

```bash
chroma> reset
```

**출력:**
```
🔄 Resetting to original search results
Original: 100 documents
✅ Filter reset complete
```

---

### 편집 명령어

#### `create <collection_name>`
새로운 콜렉션을 생성합니다.

```bash
chroma> create my_new_collection
```

**출력:**
```
✅ Collection 'my_new_collection' created successfully
```

---

#### `add <collection_name> <content> [--id <doc_id>] [--meta key=val ...]`
콜렉션에 새 문서를 추가합니다.

```bash
# 기본 (ID 자동 생성)
chroma> add my_docs 'Python is a great programming language'

# 커스텀 ID 지정
chroma> add my_docs 'Python tutorial' --id tutorial_001

# 메타데이터 포함
chroma> add tech_docs 'AI article' --meta category=tech author=John year=2024

# 메타데이터 여러 개
chroma> add my_docs 'Data science guide' --id ds_001 --meta category=tech difficulty=intermediate topic=ml
```

**매개변수:**
- `collection_name`: 콜렉션 이름 (필수)
- `content`: 문서 내용 (필수, 여러 단어는 따옴표로 감싸기)
- `--id <doc_id>`: 문서 ID (선택, 미지정 시 자동 생성)
- `--meta key=val`: 메타데이터 (선택, 여러 개 가능)

**자동 생성되는 메타데이터:**
- `added_by`: "cli"
- `timestamp`: 현재 시간

**출력:**
```
✅ Document added successfully
   Collection: tech_docs
   ID: tutorial_001
   Metadata: {'added_by': 'cli', 'timestamp': '2024-...', 'category': 'tech', 'author': 'John'}
   Content: AI article
```

---

#### `delete <collection_name> <doc_id>`
콜렉션에서 특정 문서를 삭제합니다.

```bash
chroma> delete my_collection doc_001
```

**출력:**
```
✅ Document 'doc_001' deleted from 'my_collection'
```

---

#### `drop <collection_name>`
전체 콜렉션을 삭제합니다 (확인 필요).

```bash
chroma> drop my_old_collection
```

**확인 프롬프트:**
```
⚠️  Are you sure you want to delete 'my_old_collection'? (yes/no): yes
✅ Collection 'my_old_collection' deleted successfully
```

> ⚠️ **경고**: 이 명령은 콜렉션의 모든 문서를 삭제합니다!

---

### 유틸리티 명령어

#### `health`
ChromaDB 연결 상태를 확인합니다.

```bash
chroma> health
```

**출력:**
```
✅ ChromaDB is healthy
   Heartbeat: 12345678
```

---

#### `history`
최근 명령어 히스토리를 출력합니다 (최대 20개).

```bash
chroma> history
```

**출력:**
```
============================================================
Command History (last 20)
============================================================
    1  collections
    2  info stock_knowledge
    3  search stock_knowledge "Tesla" 50
    4  filter 0.5
    5  metadata category tech
   ...
============================================================
Total: 20 commands
Use ↑↓ arrows to navigate history
```

---

#### `clear`
화면을 지우고 환영 메시지를 다시 표시합니다.

```bash
chroma> clear
```

---

#### `help`
도움말을 출력합니다.

```bash
chroma> help
```

---

#### `exit` / `quit`
프로그램을 종료합니다.

```bash
chroma> exit
chroma> quit
```

**출력:**
```
👋 Goodbye!
```

---

## 🎯 실전 사용 예시

### 예시 1: 파일에서 문서 임포트
```bash
$ python test_chroma_util.py ./chroma-data

# 1. 파일 미리보기
chroma> preview ./sample_documents.py
📄 File Preview: ./sample_documents.py
============================================================
Total documents: 8
...

# 2. 콜렉션 생성
chroma> create code_samples
✅ Collection 'code_samples' created successfully

# 3. 파일 임포트
chroma> import code_samples ./sample_documents.py
📂 Loading documents from: ./sample_documents.py
✅ Loaded 8 documents from file
✅ Prepared 8 documents for import
   - Auto-generated IDs: 2
   - Custom IDs: 6

📥 Importing documents (batch_size=100)...
   Batch 1: 8 documents ✓

============================================================
✅ Import completed
   Total: 8
   Imported: 8
============================================================

# 4. 임포트된 문서 확인
chroma> info code_samples
📖 Collection: code_samples
   Documents: 8

chroma> show code_samples 0 3
# 처음 3개 문서 출력

# 5. 검색 테스트
chroma> search code_samples "Java Spring Service" 5
🔍 Searching for: 'Java Spring Service'
# 관련 Java 코드 문서 출력
```

---

### 예시 2: 기본 탐색
```bash
$ python test_chroma_util.py ./stock-rag-data

chroma> collections
📚 Collections:
  1. stock_knowledge (1234 documents)

chroma> info stock_knowledge
📖 Collection: stock_knowledge
   Documents: 1234

chroma> show stock_knowledge 0 5
# 처음 5개 문서 출력
```

---

### 예시 3: 검색 및 필터링 (AND 조건)
```bash
chroma> search stock_knowledge "Tesla stock analysis" 100
🔍 Searching for: 'Tesla stock analysis'
# 100개 결과 반환

chroma> filter 0.7
🔎 Filtering current results (similarity >= 0.7)
Before: 100 documents
After filter: 45 documents
# 유사도 0.7 이상인 45개

chroma> metadata source analyst_report
🔎 Filtering current results by metadata: source=analyst_report
Before: 45 documents
After filter: 12 documents
# source=analyst_report인 12개

chroma> top 5
🏆 Top 5 documents by similarity
Before: 12 documents
After top: 5 documents
# 최종적으로 상위 5개만 표시

chroma> reset
🔄 Resetting to original search results
Original: 100 documents
✅ Filter reset complete
# 다시 100개로 복원
```

---

### 예시 4: 문서 추가 및 검색
```bash
chroma> create test_collection
✅ Collection 'test_collection' created successfully

chroma> add test_collection 'Python is a versatile programming language' --meta category=programming difficulty=beginner
✅ Document added successfully
   Collection: test_collection
   ID: doc_a3f8b912

chroma> add test_collection 'Machine learning with PyTorch' --id ml_001 --meta category=ai difficulty=advanced framework=pytorch
✅ Document added successfully
   Collection: test_collection
   ID: ml_001

chroma> search test_collection "Python programming" 10
🔍 Searching for: 'Python programming'
# 관련 문서 검색

chroma> delete test_collection doc_a3f8b912
✅ Document 'doc_a3f8b912' deleted from 'test_collection'

chroma> drop test_collection
⚠️  Are you sure you want to delete 'test_collection'? (yes/no): yes
✅ Collection 'test_collection' deleted successfully
```

---

### 예시 5: 복잡한 필터링 체인
```bash
# 시나리오: 최근 1년간 높은 평가를 받은 Python 관련 기술 문서 찾기

chroma> search tech_docs "Python best practices" 200

chroma> filter 0.6
# 유사도 0.6 이상 (예: 80개 남음)

chroma> metadata language python
# Python 관련 문서만 (예: 50개 남음)

chroma> metadata year 2024
# 2024년 문서만 (예: 30개 남음)

chroma> metadata rating excellent
# 평가가 excellent인 것만 (예: 15개 남음)

chroma> top 10
# 최종 상위 10개 표시

# 결과가 마음에 안 들면...
chroma> reset
# 처음 200개로 돌아가서 다시 시도
```

---

## ⌨️ 키보드 단축키

| 단축키 | 기능 |
|--------|------|
| `↑` / `↓` | 명령어 히스토리 탐색 |
| `TAB` | 명령어/콜렉션 이름 자동완성 |
| `Ctrl+C` | 현재 입력 취소 |
| `Ctrl+D` | 프로그램 종료 (exit와 동일) |
| `Ctrl+A` | 커서를 줄 시작으로 이동 |
| `Ctrl+E` | 커서를 줄 끝으로 이동 |
| `Ctrl+K` | 커서부터 줄 끝까지 삭제 |
| `Ctrl+U` | 커서부터 줄 시작까지 삭제 |

---

## 🏗️ 코드 구조

### 주요 클래스: `ChromaUtilCLI`

```python
class ChromaUtilCLI:
    def __init__(self, persist_directory: str, use_remote: bool):
        self.chroma = ChromaUtil(...)
        self.last_results: Optional[DocumentResults] = None      # 현재 필터링된 결과
        self.original_results: Optional[DocumentResults] = None  # 원본 검색 결과
        self.running = True
        self.history_file = "~/.chroma_util_history"
```

### 필터링 로직 (AND 조건)

```python
# 검색 - 원본 저장
def handle_search(self, args):
    results = self.chroma.search_similar(...)
    self.original_results = results  # 원본 보관
    self.last_results = results       # 현재 결과

# 필터링 - 현재 결과에 AND 조건 적용
def handle_filter(self, args):
    self.last_results = self.last_results.get_similarity_gte(min_similarity)

def handle_metadata(self, args):
    self.last_results = self.last_results.filter_by_metadata(key, value)

def handle_top(self, args):
    self.last_results = self.last_results.sort_by_similarity().limit(count)

# 리셋 - 원본으로 복원
def handle_reset(self, args):
    self.last_results = self.original_results
```

---

## 📦 의존성

### 외부 라이브러리
- `readline`: 터미널 인터랙션 (히스토리, 자동완성)
- `chromadb`: ChromaDB 클라이언트

### 프로젝트 모듈
- `rag.chroma_util.ChromaUtil`: ChromaDB 작업 래퍼 클래스
- `rag.chroma_util.DocumentResults`: 검색 결과 컨테이너 클래스

### ChromaUtil 메서드 사용
```python
chroma.show_collections()
chroma.get_collection_info(collection_name)
chroma.show_documents(collection_name, start, size)
chroma.search_similar(collection_name, query, limit)
chroma.create_collection(collection_name)
chroma.delete_collection(collection_name)
chroma.health_check()
```

### DocumentResults 메서드 사용
```python
results.get_similarity_gte(min_similarity)      # 유사도 필터링
results.filter_by_metadata(key, value)          # 메타데이터 필터링
results.sort_by_similarity(reverse=True)        # 정렬
results.limit(count)                            # 개수 제한
len(results)                                    # 문서 개수
```

---

## 🔍 히스토리 파일

명령어 히스토리는 다음 위치에 저장됩니다:
```
~/.chroma_util_history
```

- 최대 1000개의 명령어 저장
- 프로그램 종료 시 자동 저장
- 다음 실행 시 자동 로드

---

## 🐛 트러블슈팅

### 문제: `readline` 모듈을 찾을 수 없음
**해결:**
```bash
# Windows
pip install pyreadline3

# Linux/Mac (일반적으로 기본 포함)
# 추가 설치 불필요
```

---

### 문제: ChromaDB 연결 실패
**해결:**
```bash
# 1. 디렉토리 권한 확인
ls -la ./chroma-data

# 2. ChromaDB 프로세스 확인 (원격 모드)
curl http://localhost:8000/api/v1/heartbeat

# 3. 디렉토리 재생성
rm -rf ./chroma-data
python test_chroma_util.py  # 자동으로 생성됨
```

---

### 문제: TAB 자동완성이 작동하지 않음
**해결:**
- Mac의 경우 `libedit` 라이브러리 사용으로 인한 차이
- 코드에서 이미 처리되어 있음:
```python
if 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
```

---

## 💡 팁 & 트릭

### 1. 효율적인 필터링 전략
```bash
# ❌ 나쁜 예: 너무 엄격한 필터 먼저
chroma> search docs "Python" 100
chroma> filter 0.9            # 너무 적은 결과
chroma> metadata topic advanced

# ✅ 좋은 예: 점진적으로 좁히기
chroma> search docs "Python" 100
chroma> filter 0.5            # 적당한 필터
chroma> metadata topic python  # 카테고리 필터
chroma> filter 0.7            # 더 높은 품질로
chroma> top 10                # 최종 선별
```

---

### 2. 메타데이터 구조 파악
```bash
# 먼저 문서 몇 개 확인
chroma> show my_collection 0 3

# 메타데이터 키 확인 후 필터링
chroma> search my_collection "query" 50
chroma> metadata <확인한_키> <값>
```

---

### 3. 검색 결과 품질 확인
```bash
chroma> search docs "query" 50

# 유사도 범위 확인 (filter 명령 출력에 표시됨)
chroma> filter -1
# Similarity range: 0.1234 ~ 0.9876

# 적절한 임계값 설정
chroma> reset
chroma> filter 0.6
```

---

### 4. 배치 문서 추가 스크립트
CLI는 한 번에 하나의 문서만 추가하므로, 많은 문서를 추가하려면 Python 스크립트 사용:
```python
# batch_add.py
from rag.chroma_util import ChromaUtil

chroma = ChromaUtil(persist_directory="./chroma-data")
collection = chroma.client.get_collection("my_collection")

documents = [
    ("doc_1", "content 1", {"key": "value1"}),
    ("doc_2", "content 2", {"key": "value2"}),
    # ...
]

for doc_id, content, metadata in documents:
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata]
    )
```

---

## 📝 명령어 치트시트

| 명령어 | 설명 | 예시 |
|--------|------|------|
| **조회** |
| `collections` | 콜렉션 목록 | `collections` |
| `info <col>` | 콜렉션 정보 | `info my_docs` |
| `show <col> [start] [size]` | 문서 출력 | `show my_docs 0 10` |
| `search <col> <query> [limit]` | 유사도 검색 | `search my_docs "python" 50` |
| **필터링 (AND)** |
| `filter <min_sim>` | 유사도 필터 | `filter 0.5` |
| `metadata <key> <val>` | 메타데이터 필터 | `metadata category tech` |
| `top <count>` | 상위 N개 | `top 10` |
| `reset` | 필터 초기화 | `reset` |
| **파일 임포트** |
| `import <col> <file> [opts]` | 파일 임포트 | `import docs ./sample.py` |
| `preview <file> [max]` | 파일 미리보기 | `preview ./sample.py 10` |
| **편집** |
| `create <col>` | 콜렉션 생성 | `create new_docs` |
| `add <col> <content> [opts]` | 문서 추가 | `add docs "text" --id doc1` |
| `delete <col> <id>` | 문서 삭제 | `delete docs doc1` |
| `drop <col>` | 콜렉션 삭제 | `drop old_docs` |
| **유틸** |
| `health` | 연결 상태 | `health` |
| `history` | 히스토리 | `history` |
| `clear` | 화면 지우기 | `clear` |
| `help` | 도움말 | `help` |
| `exit` / `quit` | 종료 | `exit` |

---

## 🎓 학습 자료

### 관련 파일
- `rag/chroma_util.py`: ChromaUtil 클래스 구현
- `test/test_chroma_integration.py`: 통합 테스트
- `docs/CHROMADB_EXPLAINED.md`: ChromaDB 개념 설명

### 추가 학습
1. ChromaDB 공식 문서: https://docs.trychroma.com/
2. 벡터 유사도 검색: `docs/VECTOR_SIMILARITY_EXPLAINED.md`
3. RAG 워크플로우: `docs/RAG_WORKFLOW_EXPLAINED.md`

---

## 📄 라이센스

이 도구는 ollama-agents 프로젝트의 일부입니다.

---

**작성일**: 2024년 10월 1일  
**버전**: 1.0.0  
**작성자**: ollama-agents team
