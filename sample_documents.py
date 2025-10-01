#!/usr/bin/env python3
"""
샘플 문서 파일
import 명령어로 ChromaDB에 임포트할 수 있습니다.

Usage:
  chroma> import my_collection ./sample_documents.py
"""

documents = [
    {
        "id": "collect_entity_lton_place",
        "document": """
public class LtonPlace {
    private Long placeId;
    private String placeName;
    private String address;
    private Double latitude;
    private Double longitude;
    private String category;
    
    // Constructor, getters, setters
}
        """,
        "metadata": {
            "type": "entity",
            "layer": "domain",
            "package": "com.lotteon.travel.domain.collect.entity",
            "class_name": "LtonPlace",
            "language": "java"
        }
    },
    {
        "id": "collect_service_place_service",
        "document": """
@Service
public class PlaceService {
    
    @Autowired
    private PlaceRepository placeRepository;
    
    public List<LtonPlace> findNearbyPlaces(Double lat, Double lng, Double radius) {
        // Find places within radius
        return placeRepository.findByLocationNear(lat, lng, radius);
    }
    
    public LtonPlace save(LtonPlace place) {
        return placeRepository.save(place);
    }
}
        """,
        "metadata": {
            "type": "service",
            "layer": "domain",
            "package": "com.lotteon.travel.domain.collect.service",
            "class_name": "PlaceService",
            "language": "java"
        }
    },
    {
        "id": "collect_repository_place_repository",
        "document": """
@Repository
public interface PlaceRepository extends JpaRepository<LtonPlace, Long> {
    
    @Query("SELECT p FROM LtonPlace p WHERE " +
           "SQRT(POWER(p.latitude - :lat, 2) + POWER(p.longitude - :lng, 2)) < :radius")
    List<LtonPlace> findByLocationNear(
        @Param("lat") Double lat,
        @Param("lng") Double lng,
        @Param("radius") Double radius
    );
    
    List<LtonPlace> findByCategory(String category);
}
        """,
        "metadata": {
            "type": "repository",
            "layer": "infrastructure",
            "package": "com.lotteon.travel.domain.collect.repository",
            "class_name": "PlaceRepository",
            "language": "java"
        }
    },
    {
        # ID 없음 - 자동 생성됨
        "document": """
@RestController
@RequestMapping("/api/places")
public class PlaceController {
    
    @Autowired
    private PlaceService placeService;
    
    @GetMapping("/nearby")
    public ResponseEntity<List<LtonPlace>> getNearbyPlaces(
        @RequestParam Double lat,
        @RequestParam Double lng,
        @RequestParam(defaultValue = "5.0") Double radius
    ) {
        List<LtonPlace> places = placeService.findNearbyPlaces(lat, lng, radius);
        return ResponseEntity.ok(places);
    }
    
    @PostMapping
    public ResponseEntity<LtonPlace> createPlace(@RequestBody LtonPlace place) {
        LtonPlace saved = placeService.save(place);
        return ResponseEntity.ok(saved);
    }
}
        """,
        "metadata": {
            "type": "controller",
            "layer": "presentation",
            "package": "com.lotteon.travel.api.collect.controller",
            "class_name": "PlaceController",
            "language": "java"
        }
    },
    {
        "id": "python_data_processor",
        "document": """
class DataProcessor:
    '''데이터 처리 유틸리티'''
    
    def __init__(self, config):
        self.config = config
        
    def process(self, data):
        '''데이터 처리'''
        # 데이터 정제
        cleaned = self.clean(data)
        
        # 데이터 변환
        transformed = self.transform(cleaned)
        
        # 데이터 검증
        validated = self.validate(transformed)
        
        return validated
    
    def clean(self, data):
        # 중복 제거, null 처리
        pass
    
    def transform(self, data):
        # 형식 변환
        pass
    
    def validate(self, data):
        # 유효성 검사
        pass
        """,
        "metadata": {
            "type": "utility",
            "layer": "application",
            "package": "utils.processor",
            "class_name": "DataProcessor",
            "language": "python",
            "description": "Data processing utility class"
        }
    },
    {
        "document": """
def fibonacci(n):
    '''피보나치 수열 계산
    
    Args:
        n: 계산할 항의 개수
        
    Returns:
        피보나치 수열 리스트
        
    Examples:
        >>> fibonacci(5)
        [0, 1, 1, 2, 3]
    '''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib
        """,
        "metadata": {
            "type": "function",
            "category": "algorithm",
            "difficulty": "easy",
            "language": "python",
            "tags": ["fibonacci", "recursion", "dynamic-programming"]
        }
    },
    {
        "id": "react_component_button",
        "document": """
import React from 'react';
import './Button.css';

interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  variant = 'primary',
  disabled = false
}) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
};
        """,
        "metadata": {
            "type": "component",
            "framework": "react",
            "language": "typescript",
            "component_name": "Button",
            "category": "ui"
        }
    },
    {
        "id": "sql_query_user_stats",
        "document": """
-- 사용자별 활동 통계 조회
SELECT 
    u.user_id,
    u.username,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order_date,
    DATEDIFF(CURRENT_DATE, MAX(o.order_date)) as days_since_last_order
FROM 
    users u
    LEFT JOIN orders o ON u.user_id = o.user_id
WHERE 
    u.created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
GROUP BY 
    u.user_id, u.username
HAVING 
    total_orders > 0
ORDER BY 
    total_spent DESC
LIMIT 100;
        """,
        "metadata": {
            "type": "query",
            "language": "sql",
            "database": "mysql",
            "purpose": "analytics",
            "category": "user-statistics",
            "complexity": "medium"
        }
    }
]
