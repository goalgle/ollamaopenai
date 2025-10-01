# 📥 파일 임포트 가이드

ChromaDB에 여러 문서를 한번에 임포트하는 방법을 설명합니다.

---

## 📋 개요

`import` 명령어를 사용하면 Python 파일로부터 여러 문서를 ChromaDB에 일괄 임포트할 수 있습니다.

### 주요 기능
- ✅ 여러 문서 일괄 임포트
- ✅ 자동 ID 생성 또는 커스텀 ID 사용
- ✅ 메타데이터 포함
- ✅ 배치 처리로 대량 데이터 지원
- ✅ 임포트 전 파일 미리보기

---

## 🚀 빠른 시작

### 1단계: 문서 파일 준비

`my_documents.py` 파일을 생성합니다:

```python
documents = [
    {
        "id": "doc_001",           # Optional: 생략 시 자동 생성
        "document": "문서 내용",    # Required
        "metadata": {              # Optional
            "category": "tech",
            "author": "John"
        }
    },
    {
        "document": "또 다른 문서",  # ID 없음 - 자동 생성됨
        "metadata": {
            "category": "news"
        }
    }
]
```

### 2단계: 파일 미리보기 (선택)

```bash
chroma> preview ./my_documents.py
```

### 3단계: 콜렉션 생성

```bash
chroma> create my_collection
```

### 4단계: 문서 임포트

```bash
chroma> import my_collection ./my_documents.py
```

---

## 📝 파일 포맷 상세

### 기본 구조

```python
documents = [
    {
        "id": str,          # Optional: 문서 ID (생략 시 자동 생성)
        "document": str,    # Required: 문서 내용
        "metadata": dict    # Optional: 메타데이터
    },
    ...
]
```

### 필드 설명

#### `id` (선택)
- **타입**: 문자열
- **설명**: 문서의 고유 식별자
- **생략 가능**: `--no-auto-id` 옵션이 없으면 자동 생성
- **주의**: 중복된 ID는 허용되지 않음

```python
{
    "id": "user_guide_001",  # 명시적 ID
    "document": "..."
}

{
    # ID 없음 - doc_a3f8b912 같은 형태로 자동 생성
    "document": "..."
}
```

#### `document` (필수)
- **타입**: 문자열
- **설명**: 문서의 실제 내용
- **필수**: 모든 문서에 반드시 포함되어야 함

```python
{
    "document": "This is the document content."
}

# 멀티라인도 가능
{
    "document": """
    This is a multi-line
    document content.
    It can span multiple lines.
    """
}
```

#### `metadata` (선택)
- **타입**: 딕셔너리
- **설명**: 문서의 메타데이터
- **생략 가능**: 빈 딕셔너리로 처리됨

```python
{
    "document": "...",
    "metadata": {
        "category": "tutorial",
        "language": "python",
        "difficulty": "beginner",
        "tags": ["basics", "intro"],
        "year": 2024
    }
}
```

---

## ⚙️ 명령어 옵션

### 기본 사용법

```bash
import <collection_name> <file_path>
```

### 옵션

#### `--no-auto-id`
ID 자동 생성을 비활성화합니다.

```bash
chroma> import my_docs ./docs.py --no-auto-id
```

#### `--batch-size <size>`
배치 크기 지정 (기본값: 100)

```bash
chroma> import my_docs ./docs.py --batch-size 50
```

---

## 🎯 사용 예시

### 예시 1: 코드 스니펫 임포트

```python
# code_snippets.py
documents = [
    {
        "id": "py_list_comprehension",
        "document": """
squares = [x**2 for x in range(10)]
        """,
        "metadata": {
            "language": "python",
            "topic": "list-comprehension"
        }
    }
]
```

### 예시 2: 대량 데이터

```python
# large_dataset.py
documents = []
for i in range(500):
    documents.append({
        "id": f"article_{i:04d}",
        "document": f"Article {i} content",
        "metadata": {"index": i}
    })
```

---

## 💡 팁 & 베스트 프랙티스

### 1. 메타데이터 활용

```python
# 좋은 예
{
    "document": "...",
    "metadata": {
        "type": "tutorial",
        "language": "python",
        "level": "beginner"
    }
}
```

### 2. ID 관리

```python
# 자동 생성 (권장)
{"document": "..."}

# 커스텀 ID
{"id": "user_guide_001", "document": "..."}
```

---

## 🐛 문제 해결

### "File not found"
```bash
# 절대 경로 사용
chroma> import docs /absolute/path/to/docs.py
```

### "Collection not found"
```bash
# 콜렉션 먼저 생성
chroma> create my_docs
chroma> import my_docs ./docs.py
```

### "Duplicate document ID"
```python
# ID를 고유하게 하거나 자동 생성 사용
documents = [
    {"document": "First"},   # 자동 생성
    {"document": "Second"}   # 자동 생성
]
```

---

**Happy Importing! 📥**
