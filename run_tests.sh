#!/bin/bash
# 테스트 실행 스크립트

set -e  # 에러 발생 시 중단

echo "============================================"
echo "ollama-agents 테스트 실행"
echo "============================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 인자 파싱
TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
  "unit")
    echo -e "${YELLOW}[단위 테스트]${NC} Mock을 사용한 빠른 테스트 실행..."
    echo ""
    pytest -m unit -v
    ;;
    
  "integration")
    echo -e "${YELLOW}[통합 테스트]${NC} ChromaDB 연결 확인 중..."
    
    # ChromaDB 연결 확인
    if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} ChromaDB 연결 성공"
        echo ""
        echo -e "${YELLOW}[통합 테스트]${NC} 실제 ChromaDB를 사용한 테스트 실행..."
        echo ""
        pytest -m integration -v
    else
        echo -e "${RED}✗${NC} ChromaDB에 연결할 수 없습니다"
        echo ""
        echo "ChromaDB를 먼저 시작하세요:"
        echo "  docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma"
        exit 1
    fi
    ;;
    
  "chroma")
    echo -e "${YELLOW}[ChromaDB 테스트]${NC} ChromaDB 관련 테스트만 실행..."
    
    # ChromaDB 연결 확인
    if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} ChromaDB 연결 성공"
        echo ""
        pytest -m chroma -v
    else
        echo -e "${RED}✗${NC} ChromaDB에 연결할 수 없습니다"
        echo ""
        echo "ChromaDB를 먼저 시작하세요:"
        echo "  docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma"
        exit 1
    fi
    ;;
    
  "fast")
    echo -e "${YELLOW}[빠른 테스트]${NC} 통합 테스트 제외..."
    echo ""
    pytest -m "not integration" -v
    ;;
    
  "all")
    echo -e "${YELLOW}[전체 테스트]${NC} 모든 테스트 실행..."
    echo ""
    
    # 단위 테스트
    echo -e "${GREEN}[1/2]${NC} 단위 테스트..."
    pytest -m unit -v
    echo ""
    
    # 통합 테스트
    echo -e "${GREEN}[2/2]${NC} 통합 테스트..."
    if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        pytest -m integration -v
    else
        echo -e "${YELLOW}⚠${NC} ChromaDB가 실행되지 않아 통합 테스트를 건너뜁니다"
        echo "  통합 테스트를 실행하려면:"
        echo "  docker run -d --name chromadb -v ./chroma-data:/chroma/chroma -p 8000:8000 chromadb/chroma"
    fi
    ;;
    
  "coverage")
    echo -e "${YELLOW}[커버리지 테스트]${NC} 코드 커버리지 측정..."
    echo ""
    pytest --cov=rag --cov-report=html --cov-report=term-missing
    echo ""
    echo -e "${GREEN}✓${NC} HTML 리포트 생성: htmlcov/index.html"
    ;;
    
  *)
    echo "사용법: $0 [unit|integration|chroma|fast|all|coverage]"
    echo ""
    echo "옵션:"
    echo "  unit        - 단위 테스트만 (Mock, 빠름)"
    echo "  integration - 통합 테스트만 (ChromaDB 필요)"
    echo "  chroma      - ChromaDB 테스트만"
    echo "  fast        - 통합 테스트 제외"
    echo "  all         - 전체 테스트 (기본값)"
    echo "  coverage    - 커버리지 측정"
    echo ""
    echo "예시:"
    echo "  $0 unit"
    echo "  $0 integration"
    echo "  $0 coverage"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}테스트 완료!${NC}"
echo -e "${GREEN}============================================${NC}"
