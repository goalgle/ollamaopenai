# 🧮 벡터 유사도 검색의 원리

Vector Similarity Search가 어떻게 작동하는지 쉽게 설명합니다.

---

## 🎯 핵심 아이디어

```
텍스트(문자)는 컴퓨터가 계산할 수 없음
    ↓
숫자(벡터)로 변환
    ↓
수학적으로 유사도 계산 가능!
```

---

## 📊 STEP 1: 텍스트를 벡터로 변환

### 간단한 예시 (7차원)

```
키워드: [python, java, programming, food, cooking, music, sports]

문서1: "Python is a programming language"
벡터1: [1, 0, 1, 0, 0, 0, 0]
        ↑     ↑
     python programming

문서2: "I love cooking Italian food"
벡터2: [0, 0, 0, 1, 1, 0, 0]
                 ↑  ↑
              food cooking
```

### 시각화

```
문서1 (Python programming):
  python      : ██████████ (1.0)
  java        :            (0.0)
  programming : ██████████ (1.0)
  food        :            (0.0)
  cooking     :            (0.0)
  music       :            (0.0)
  sports      :            (0.0)

문서2 (Italian food):
  python      :            (0.0)
  java        :            (0.0)
  programming :            (0.0)
  food        : ██████████ (1.0)
  cooking     : ██████████ (1.0)
  music       :            (0.0)
  sports      :            (0.0)
```

---

## 📐 STEP 2: 유사도 계산

### 코사인 유사도 (Cosine Similarity)

**개념:** 두 벡터 사이의 각도 측정

```
      벡터A
       ↑
       |  θ (각도)
       | ↗
       |↗
       +--------→ 벡터B

코사인 유사도 = cos(θ)
```

**값의 의미:**
- `1.0` = 0도 = 완전히 같은 방향 (매우 유사) ✅
- `0.5` = 60도 = 어느 정도 유사 🟡
- `0.0` = 90도 = 직각 (무관) ⚪
- `-1.0` = 180도 = 반대 방향 ❌

### 계산 예시

```python
벡터1 = [1, 0, 1, 0, 0, 0, 0]  # Python programming
벡터2 = [1, 0, 1, 0, 0, 0, 0]  # Python programming (같음)
벡터3 = [0, 1, 1, 0, 0, 0, 0]  # Java programming
벡터4 = [0, 0, 0, 1, 1, 0, 0]  # Italian food

유사도(벡터1, 벡터2) = 1.00  ← 완전히 같음
유사도(벡터1, 벡터3) = 0.50  ← programming 공통
유사도(벡터1, 벡터4) = 0.00  ← 완전히 다름
```

---

## 🔍 STEP 3: 검색 작동 원리

### 검색 과정

```
[1] 질문: "Python programming tutorial"
    ↓
[2] 질문을 벡터로 변환
    쿼리벡터 = [1, 0, 1, 0, 0, 0, 0]
    ↓
[3] DB의 모든 문서 벡터와 유사도 계산
    문서1: 유사도 0.85 ← Python programming 🟢
    문서2: 유사도 0.82 ← Python coding    🟢
    문서3: 유사도 0.45 ← Java programming  🟡
    문서4: 유사도 0.10 ← Cooking recipes   🔴
    ↓
[4] 유사도 높은 순으로 정렬
    [문서1, 문서2, 문서3, ...]
    ↓
[5] 상위 N개 반환
    Top 3: [문서1, 문서2, 문서3]
```

### 코드로 보기

```python
# 1. 질문을 벡터로
query = "Python programming"
query_vector = embedding_model.encode(query)  # [0.1, 0.3, 0.8, ...]

# 2. 유사도 계산
similarities = []
for doc in documents:
    sim = cosine_similarity(query_vector, doc.vector)
    similarities.append((doc, sim))

# 3. 정렬 후 반환
similarities.sort(key=lambda x: x[1], reverse=True)
return similarities[:5]  # 상위 5개
```

---

## ⚡ STEP 4: 실제 임베딩의 위력

### 간단한 방법 vs 실제 임베딩

| 항목 | 간단한 방법 | 실제 임베딩 (BERT, OpenAI) |
|------|------------|---------------------------|
| **차원** | 7차원 | 384~1536차원 |
| **키워드 매칭** | ✅ | ✅ |
| **동의어 인식** | ❌ | ✅ |
| **문맥 이해** | ❌ | ✅ |
| **다국어** | ❌ | ✅ |

### 실제 임베딩의 마법 ✨

```
질문: "투자 방법"

간단한 방법:
  → "투자"만 찾음 ❌

실제 임베딩:
  → "투자" ✅
  → "재테크" ✅
  → "자산 관리" ✅
  → "포트폴리오" ✅
  → "investment" ✅ (영어도!)
  
→ 모두 비슷한 벡터로 변환되어 자동으로 찾아짐!
```

---

## 🎨 시각적 이해

### 2차원으로 단순화한 벡터 공간

```
         프로그래밍
              ↑
              |
    Python •  |  • Java
              |
    Ruby •    |    • C++
              |
──────────────+────────────→ 언어 종류
              |
              |  • 김치찌개
              |
    피자 •    |    • 파스타
              |
           음식
```

**관찰:**
- Python, Java, Ruby, C++ = 모두 프로그래밍 영역에 밀집
- 김치찌개, 피자, 파스타 = 모두 음식 영역에 밀집
- 프로그래밍 ↔ 음식 = 멀리 떨어짐

### 검색 예시

```
질문: "프로그래밍 언어"를 벡터로 변환하면
→ 프로그래밍 영역의 벡터가 됨

가장 가까운 문서 찾기:
  1. Python   (거리: 0.1) ← 매우 가까움!
  2. Java     (거리: 0.2) ← 가까움
  3. Ruby     (거리: 0.3) ← 가까움
  ...
  99. 김치찌개 (거리: 9.8) ← 매우 멀음
```

---

## 🧪 실제 ChromaDB의 동작

### 문서 저장 과정

```
[1] 사용자가 문서 저장 요청
    "Python은 1991년에 개발되었다"
    ↓
[2] 임베딩 모델로 변환
    → [0.12, 0.45, 0.89, ..., 0.34] (384차원)
    ↓
[3] ChromaDB에 저장
    {
      id: "doc_001",
      vector: [0.12, 0.45, ...],
      content: "Python은 1991년에...",
      metadata: {topic: "history"}
    }
```

### 검색 과정

```
[1] 사용자 질문
    "파이썬의 역사"
    ↓
[2] 질문을 벡터로 변환
    → [0.13, 0.44, 0.88, ..., 0.35] (384차원)
    ↓
[3] DB의 모든 벡터와 유사도 계산
    ANN (Approximate Nearest Neighbor) 알고리즘 사용
    ↓
[4] 상위 N개 반환
    [
      {id: "doc_001", similarity: 0.95, ...},
      {id: "doc_015", similarity: 0.87, ...},
      ...
    ]
```

---

## 💡 왜 이게 혁명적인가?

### 전통 검색의 한계

```sql
-- SQL 검색
SELECT * FROM docs WHERE content LIKE '%투자%';

문제점:
❌ "투자"라는 단어가 정확히 있어야만 검색됨
❌ "재테크", "자산관리"는 못 찾음
❌ 오타나 다른 표현은 못 찾음
❌ 영어/한글 혼용 어려움
```

### 벡터 검색의 장점

```python
# 벡터 검색
search("투자 방법")

장점:
✅ "투자" 뿐만 아니라
✅ "재테크", "자산관리", "포트폴리오"도 찾음
✅ "investment" (영어)도 찾음
✅ 약간의 오타도 괜찮음
✅ 의미가 같으면 표현이 달라도 OK
```

---

## 🔬 고급 개념

### 1. ANN (Approximate Nearest Neighbor)

수백만 개 문서에서 정확한 유사도를 모두 계산하면 너무 느림!

**해결책:** 근사 알고리즘
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- LSH (Locality Sensitive Hashing)

```
정확한 검색: 1,000,000개 모두 계산 → 10초
근사 검색: 상위 후보만 계산 → 0.01초 (정확도 99%)
```

### 2. 다차원 공간

```
우리가 보는 세상: 3차원 (x, y, z)
임베딩 공간: 384~1536차원!

상상할 수 없지만, 수학적으로는 완벽하게 작동
→ 더 많은 차원 = 더 미묘한 의미 차이 포착 가능
```

### 3. 학습된 임베딩

```
임베딩 모델은 수십억 개의 문서로 학습됨

결과:
- "king" - "man" + "woman" ≈ "queen"
- "Paris" - "France" + "Italy" ≈ "Rome"
- "투자" ≈ "investment" (다국어!)

→ 단어 간의 관계를 자동으로 학습!
```

---

## 🎯 실전 예제

### 예제 1: 비슷한 질문 찾기

```
질문 1: "Python으로 리스트 정렬하는 법"
질문 2: "파이썬 배열 sort 방법"
질문 3: "리스트를 순서대로 배열하기"

→ 모두 비슷한 벡터!
→ 중복 질문 자동 감지 가능
```

### 예제 2: 추천 시스템

```
사용자가 읽은 문서:
"Python 튜토리얼"의 벡터 = [0.1, 0.5, 0.8, ...]

비슷한 벡터 찾기:
1. "Django 입문" (유사도 0.85)
2. "FastAPI 가이드" (유사도 0.82)
3. "Flask 시작하기" (유사도 0.79)

→ 자동 추천!
```

### 예제 3: 다국어 검색

```
한글 질문: "기계 학습 알고리즘"
벡터: [0.2, 0.6, 0.9, ...]

영어 문서:
- "Machine Learning Algorithms" (유사도 0.93) ✅
- "ML Tutorial" (유사도 0.87) ✅

→ 번역 없이 자동으로 찾아줌!
```

---

## 📚 더 알아보기

### 실습

```bash
# 1. 원리 이해
python examples/vector_similarity_explained.py

# 2. 실제 검색 테스트
python examples/test_similarity_search.py

# 3. 문서 요약
python examples/document_summarization_example.py
```

### 주요 임베딩 모델

| 모델 | 차원 | 특징 |
|------|------|------|
| **OpenAI text-embedding-3-large** | 1536 | 최고 성능, 유료 |
| **OpenAI text-embedding-3-small** | 512 | 빠름, 유료 |
| **BERT** | 768 | 무료, 범용 |
| **Sentence-BERT** | 384 | 빠름, 무료 |
| **multilingual-e5** | 384 | 다국어 특화 |

---

## 🎉 정리

**벡터 유사도 검색 = 텍스트를 숫자로 바꿔서 수학으로 유사도 계산**

```
텍스트 → 벡터(숫자) → 유사도 계산 → 검색 결과
```

**핵심 장점:**
- ✅ 키워드 없이도 의미로 검색
- ✅ 동의어 자동 인식
- ✅ 다국어 지원
- ✅ 문맥 이해

**활용:**
- 🔍 검색 엔진
- 🤖 RAG 시스템
- 📝 문서 분류/요약
- 💬 챗봇
- 🎯 추천 시스템

---

**실제로 체험해보세요!**
```bash
python examples/vector_similarity_explained.py
```
