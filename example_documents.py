# ChromaDB Import Example File
# Usage: import <collection_name> example_documents.py

documents = [
    {
        "id": "collect_entity_lton_place",
        "document": """클래스명: LtonPlace
패키지: com.lotteon.travel.domain.collect.entity
파일경로: src/main/java/com/lotteon/travel/domain/collect/entity/LtonPlace.java

설명:
- 장소 정보를 관리하는 엔티티 클래스
- JPA를 사용하여 데이터베이스와 매핑
- lton_place 테이블과 연동

주요 필드:
- id: 장소 ID (Primary Key)
- name: 장소명
- address: 주소
- latitude: 위도
- longitude: 경도""",
        "metadata": {
            "type": "entity",
            "layer": "domain",
            "package": "com.lotteon.travel.domain.collect.entity",
            "class_name": "LtonPlace",
            "framework": "JPA",
            "database_table": "lton_place",
            "related_classes": "LtonPlaceRepository,LtonPlaceService"
        }
    },
    {
        "id": "collect_repository_lton_place",
        "document": """클래스명: LtonPlaceRepository
패키지: com.lotteon.travel.domain.collect.repository
파일경로: src/main/java/com/lotteon/travel/domain/collect/repository/LtonPlaceRepository.java

설명:
- LtonPlace 엔티티의 데이터 접근 계층
- Spring Data JPA Repository 인터페이스
- 기본 CRUD 작업 제공

주요 메서드:
- findById(Long id): ID로 장소 조회
- findAll(): 모든 장소 조회
- save(LtonPlace entity): 장소 저장
- delete(LtonPlace entity): 장소 삭제""",
        "metadata": {
            "type": "repository",
            "layer": "persistence",
            "package": "com.lotteon.travel.domain.collect.repository",
            "class_name": "LtonPlaceRepository",
            "framework": "Spring Data JPA",
            "extends": "JpaRepository"
        }
    },
    {
        "id": "collect_service_lton_place",
        "document": """클래스명: LtonPlaceService
패키지: com.lotteon.travel.domain.collect.service
파일경로: src/main/java/com/lotteon/travel/domain/collect/service/LtonPlaceService.java

설명:
- LtonPlace 관련 비즈니스 로직 처리
- Repository를 통한 데이터 접근
- 트랜잭션 관리

주요 메서드:
- getPlaceById(Long id): 장소 정보 조회
- createPlace(PlaceDto dto): 장소 생성
- updatePlace(Long id, PlaceDto dto): 장소 정보 수정
- deletePlace(Long id): 장소 삭제
- searchPlaces(SearchCriteria criteria): 장소 검색""",
        "metadata": {
            "type": "service",
            "layer": "service",
            "package": "com.lotteon.travel.domain.collect.service",
            "class_name": "LtonPlaceService",
            "framework": "Spring",
            "dependencies": "LtonPlaceRepository"
        }
    },
    {
        # ID 없음 - auto-id 테스트용
        "document": """클래스명: LtonPlaceController
패키지: com.lotteon.travel.domain.collect.controller
파일경로: src/main/java/com/lotteon/travel/domain/collect/controller/LtonPlaceController.java

설명:
- 장소 관련 REST API 엔드포인트
- 클라이언트 요청 처리 및 응답
- HTTP 요청/응답 변환

주요 엔드포인트:
- GET /api/places: 장소 목록 조회
- GET /api/places/{id}: 장소 상세 조회
- POST /api/places: 새 장소 생성
- PUT /api/places/{id}: 장소 정보 수정
- DELETE /api/places/{id}: 장소 삭제""",
        "metadata": {
            "type": "controller",
            "layer": "presentation",
            "package": "com.lotteon.travel.domain.collect.controller",
            "class_name": "LtonPlaceController",
            "framework": "Spring MVC",
            "dependencies": "LtonPlaceService",
            "base_path": "/api/places"
        }
    }
]
