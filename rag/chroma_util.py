#!/usr/bin/env python3
"""
ChromaDB Utility Class
ChromaDB에 쉽게 접근할 수 있는 유틸리티 클래스

사용 예시:
    # 콜렉션 목록 출력
    chroma = ChromaUtil()
    chroma.show_collections()
    
    # 문서 출력 (0번부터 10개)
    chroma.show_documents("my_collection", 0, 10)
    
    # 유사도 필터링
    chroma.show_documents("my_collection", 0, 10).get_similarity_gte(0.5)
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """문서 검색 결과를 담는 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float = 1.0


class DocumentResults:
    """
    문서 검색 결과 컬렉션 (Fluent API 지원)
    
    체이닝 메서드를 제공하여 결과를 필터링하고 조작할 수 있습니다.
    """
    
    def __init__(self, results: List[DocumentResult]):
        self.results = results
    
    def get_similarity_gte(self, threshold: float) -> 'DocumentResults':
        """
        유사도가 threshold 이상인 문서만 필터링
        
        Args:
            threshold: 최소 유사도 (0.0 ~ 1.0)
            
        Returns:
            필터링된 DocumentResults
        """
        filtered = [r for r in self.results if r.similarity_score >= threshold]
        return DocumentResults(filtered)
    
    def get_similarity_lte(self, threshold: float) -> 'DocumentResults':
        """
        유사도가 threshold 이하인 문서만 필터링
        
        Args:
            threshold: 최대 유사도 (0.0 ~ 1.0)
            
        Returns:
            필터링된 DocumentResults
        """
        filtered = [r for r in self.results if r.similarity_score <= threshold]
        return DocumentResults(filtered)
    
    def filter_by_metadata(self, key: str, value: Any) -> 'DocumentResults':
        """
        메타데이터로 필터링
        
        Args:
            key: 메타데이터 키
            value: 필터링할 값
            
        Returns:
            필터링된 DocumentResults
        """
        filtered = [
            r for r in self.results 
            if r.metadata.get(key) == value
        ]
        return DocumentResults(filtered)
    
    def sort_by_similarity(self, reverse: bool = True) -> 'DocumentResults':
        """
        유사도로 정렬
        
        Args:
            reverse: True면 내림차순, False면 오름차순
            
        Returns:
            정렬된 DocumentResults
        """
        sorted_results = sorted(
            self.results, 
            key=lambda r: r.similarity_score, 
            reverse=reverse
        )
        return DocumentResults(sorted_results)
    
    def limit(self, count: int) -> 'DocumentResults':
        """
        결과 개수 제한
        
        Args:
            count: 최대 결과 개수
            
        Returns:
            제한된 DocumentResults
        """
        return DocumentResults(self.results[:count])
    
    def to_list(self) -> List[DocumentResult]:
        """결과를 리스트로 반환"""
        return self.results
    
    def __len__(self) -> int:
        """결과 개수"""
        return len(self.results)
    
    def __iter__(self):
        """이터레이터"""
        return iter(self.results)
    
    def __repr__(self) -> str:
        """문자열 표현"""
        if not self.results:
            return "DocumentResults(empty)"
        
        lines = [f"DocumentResults(count={len(self.results)})"]
        for i, result in enumerate(self.results, 1):
            content_preview = result.content[:50] + "..." if len(result.content) > 50 else result.content
            lines.append(
                f"  [{i}] ID: {result.id}, "
                f"Similarity: {result.similarity_score:.4f}, "
                f"Content: {content_preview}"
            )
        return "\n".join(lines)


class ChromaUtil:
    """
    ChromaDB 쉬운 접근 유틸리티
    
    사용 예시:
        >>> chroma = ChromaUtil()
        >>> chroma.show_collections()
        >>> chroma.show_documents("my_collection", 0, 10)
        >>> chroma.show_documents("my_collection").get_similarity_gte(0.5)
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma-data",
        host: str = "localhost",
        port: int = 8000,
        use_remote: bool = True
    ):
        """
        ChromaUtil 초기화
        
        Args:
            persist_directory: 로컬 모드에서 데이터 저장 경로
            host: 원격 ChromaDB 서버 호스트
            port: 원격 ChromaDB 서버 포트
            use_remote: True면 원격 서버 사용, False면 로컬 모드
        """
        self.use_remote = use_remote
        
        if use_remote:
            logger.info(f"Connecting to remote ChromaDB at {host}:{port}")
            try:
                self.client = chromadb.HttpClient(host=host, port=port)
            except Exception as e:
                logger.warning(f"Failed to connect to remote ChromaDB: {e}")
                logger.info(f"Falling back to local mode at {persist_directory}")
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.use_remote = False
        else:
            logger.info(f"Using local ChromaDB at {persist_directory}")
            self.client = chromadb.PersistentClient(path=persist_directory)
    
    def show_collections(self) -> List[str]:
        """
        모든 콜렉션 목록을 출력하고 반환
        
        Returns:
            콜렉션 이름 리스트
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            print(f"\n{'='*60}")
            print(f"Total Collections: {len(collection_names)}")
            print(f"{'='*60}")
            
            for i, name in enumerate(collection_names, 1):
                # 각 콜렉션의 문서 개수도 표시
                try:
                    collection = self.client.get_collection(name=name)
                    count = collection.count()
                    metadata = collection.metadata
                    
                    print(f"\n[{i}] Collection: {name}")
                    print(f"    Documents: {count}")
                    if metadata:
                        print(f"    Metadata: {metadata}")
                except Exception as e:
                    print(f"\n[{i}] Collection: {name}")
                    print(f"    Error getting details: {e}")
            
            print(f"\n{'='*60}\n")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            print(f"Error: {e}")
            return []
    
    def show_documents(
        self,
        collection_name: str,
        start: int = 0,
        size: int = 10,
        query_text: Optional[str] = None
    ) -> DocumentResults:
        """
        컬렉션의 문서를 출력하고 DocumentResults 반환
        
        Args:
            collection_name: 콜렉션 이름
            start: 시작 인덱스
            size: 가져올 문서 개수
            query_text: 검색할 텍스트 (없으면 전체 문서 조회)
            
        Returns:
            DocumentResults 객체 (체이닝 가능)
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            total_count = collection.count()
            
            print(f"\n{'='*60}")
            print(f"Collection: {collection_name}")
            print(f"Total Documents: {total_count}")
            print(f"Showing: {start} ~ {start + size - 1}")
            print(f"{'='*60}\n")
            
            # query_text가 있으면 검색, 없으면 전체 조회
            if query_text:
                # 쿼리 검색 (임베딩 필요)
                results = collection.query(
                    query_texts=[query_text],
                    n_results=size
                )
                
                document_results = []
                if results['ids'] and len(results['ids']) > 0:
                    for i in range(len(results['ids'][0])):
                        # ChromaDB의 distance를 similarity로 변환
                        # distance가 작을수록 유사도가 높음
                        distance = results['distances'][0][i]
                        
                        # distance를 0~1 범위의 similarity로 변환
                        # cosine distance: similarity = 1 - distance
                        # 하지만 distance 값의 범위를 확인하여 적절히 변환
                        if distance < 0:
                            # 이미 음수면 코사인 유사도일 가능성
                            similarity = abs(distance)
                        elif distance <= 2.0:
                            # 일반적인 cosine distance (0~2)
                            similarity = 1.0 - (distance / 2.0)
                        else:
                            # L2 distance 등 다른 메트릭
                            # 단순히 역수를 사용 (거리가 멀수록 유사도 낮음)
                            similarity = 1.0 / (1.0 + distance)
                        
                        doc_result = DocumentResult(
                            id=results['ids'][0][i],
                            content=results['documents'][0][i],
                            metadata=results['metadatas'][0][i] or {},
                            similarity_score=similarity
                        )
                        document_results.append(doc_result)
                        
                        # 출력
                        print(f"[{i+1}] ID: {doc_result.id}")
                        print(f"    Distance: {distance:.4f}")
                        print(f"    Similarity: {doc_result.similarity_score:.4f}")
                        print(f"    Metadata: {doc_result.metadata}")
                        print(f"    Content: {doc_result.content[:200]}...")
                        print()
            else:
                # 전체 문서 조회
                # ChromaDB는 offset을 직접 지원하지 않으므로 전체 가져온 후 슬라이싱
                results = collection.get(
                    include=["documents", "metadatas"]
                )
                
                document_results = []
                if results['ids']:
                    # start부터 start+size까지 슬라이싱
                    end = min(start + size, len(results['ids']))
                    
                    for i in range(start, end):
                        doc_result = DocumentResult(
                            id=results['ids'][i],
                            content=results['documents'][i],
                            metadata=results['metadatas'][i] or {},
                            similarity_score=1.0  # 전체 조회는 유사도 없음
                        )
                        document_results.append(doc_result)
                        
                        # 출력
                        print(f"[{i-start+1}] ID: {doc_result.id}")
                        print(f"    Metadata: {doc_result.metadata}")
                        print(f"    Content: {doc_result.content[:200]}...")
                        print()
            
            print(f"{'='*60}\n")
            return DocumentResults(document_results)
            
        except Exception as e:
            logger.error(f"Failed to show documents from '{collection_name}': {e}")
            print(f"Error: {e}")
            return DocumentResults([])
    
    def search_similar(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 10
    ) -> DocumentResults:
        """
        유사도 검색 (편의 메서드)
        
        Args:
            collection_name: 콜렉션 이름
            query_text: 검색할 텍스트
            limit: 최대 결과 개수
            
        Returns:
            DocumentResults 객체
        """
        return self.show_documents(
            collection_name=collection_name,
            start=0,
            size=limit,
            query_text=query_text
        )
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        콜렉션 상세 정보 조회
        
        Args:
            collection_name: 콜렉션 이름
            
        Returns:
            콜렉션 정보 딕셔너리
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            info = {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            
            print(f"\n{'='*60}")
            print(f"Collection: {info['name']}")
            print(f"Documents: {info['count']}")
            print(f"Metadata: {info['metadata']}")
            print(f"{'='*60}\n")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            print(f"Error: {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        콜렉션 삭제
        
        Args:
            collection_name: 삭제할 콜렉션 이름
            
        Returns:
            성공 여부
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✅ Collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        새 콜렉션 생성
        
        Args:
            collection_name: 콜렉션 이름
            metadata: 메타데이터 (None이면 메타데이터 없이 생성)
            
        Returns:
            성공 여부
        """
        try:
            # ChromaDB는 빈 딕셔너리를 허용하지 않음
            # metadata가 None이거나 빈 딕셔너리면 메타데이터 없이 생성
            if metadata:
                self.client.create_collection(
                    name=collection_name,
                    metadata=metadata
                )
            else:
                self.client.create_collection(
                    name=collection_name
                )
            print(f"✅ Collection '{collection_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        ChromaDB 연결 상태 확인
        
        Returns:
            연결 정상 여부
        """
        try:
            self.client.heartbeat()
            print("✅ ChromaDB is healthy")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            print(f"❌ ChromaDB connection failed: {e}")
            return False


# 편의 함수들
def quick_show_collections(persist_directory: str = "./chroma-data") -> List[str]:
    """빠른 콜렉션 조회"""
    util = ChromaUtil(persist_directory=persist_directory, use_remote=False)
    return util.show_collections()


def quick_show_documents(
    collection_name: str,
    start: int = 0,
    size: int = 10,
    persist_directory: str = "./chroma-data"
) -> DocumentResults:
    """빠른 문서 조회"""
    util = ChromaUtil(persist_directory=persist_directory, use_remote=False)
    return util.show_documents(collection_name, start, size)


# 예제 사용법
if __name__ == "__main__":
    # 기본 사용법
    chroma = ChromaUtil(persist_directory="./chroma-data", use_remote=False)
    
    # 콜렉션 목록
    chroma.show_collections()
    
    # 문서 조회
    results = chroma.show_documents("test_collection", 0, 10)
    
    # 체이닝으로 필터링
    filtered = results.get_similarity_gte(0.5)
    print(f"\nFiltered results (similarity >= 0.5): {len(filtered)}")
    
    # 유사도 검색
    search_results = chroma.search_similar(
        "test_collection",
        "Python programming",
        limit=5
    )
    
    # 필터링 후 정렬
    top_results = (
        search_results
        .get_similarity_gte(0.7)
        .sort_by_similarity()
        .limit(3)
    )
    print(top_results)
