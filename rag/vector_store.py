#!/usr/bin/env python3
"""
Vector Store Implementation using ChromaDB
실제 ChromaDB를 사용하는 벡터 스토어 구현
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB를 사용한 벡터 스토어 구현
    
    ChromaDB는 오픈소스 임베딩 데이터베이스로 RAG 시스템에 최적화되어 있습니다.
    - 로컬/원격 모드 모두 지원
    - 자동 영속화 (persistent storage)
    - 메타데이터 필터링 지원
    - 코사인 유사도 기반 검색
    
    사용 예시:
    ```python
    # 로컬 모드
    store = ChromaVectorStore(persist_directory="./chroma-data")
    
    # 원격 모드 (Docker)
    store = ChromaVectorStore(
        host="localhost",
        port=8000,
        use_remote=True
    )
    ```
    """

    def __init__(
        self,
        persist_directory: str = "./chroma-data",
        host: str = "localhost",
        port: int = 8000,
        use_remote: bool = True
    ):
        """
        ChromaVectorStore 초기화
        
        Args:
            persist_directory: 로컬 모드에서 데이터 저장 경로
            host: 원격 ChromaDB 서버 호스트
            port: 원격 ChromaDB 서버 포트
            use_remote: True면 원격 서버 사용, False면 로컬 모드
        """
        self.use_remote = use_remote
        
        if use_remote:
            # Docker로 실행한 ChromaDB 서버에 연결
            logger.info(f"Connecting to remote ChromaDB at {host}:{port}")
            self.client = chromadb.HttpClient(
                host=host,
                port=port
            )
        else:
            # 로컬 파일 시스템에 저장하는 모드
            logger.info(f"Using local ChromaDB at {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
        
        logger.info("ChromaDB client initialized successfully")

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        새로운 컬렉션 생성 (이미 존재하면 기존 것 사용)
        
        Args:
            collection_name: 컬렉션 이름 (예: "agent-001")
            dimension: 임베딩 벡터 차원 (ChromaDB는 자동 감지하지만 문서화 목적)
            metadata: 컬렉션 메타데이터
        
        Returns:
            bool: 성공 여부
            
        참고:
            ChromaDB는 get_or_create_collection을 사용하여
            이미 존재하면 기존 컬렉션을 반환합니다.
        """
        try:
            collection_metadata = metadata or {}
            collection_metadata["dimension"] = dimension
            collection_metadata["created_at"] = datetime.now().isoformat()
            
            # get_or_create: 없으면 생성, 있으면 기존 것 반환
            self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            
            logger.info(f"Collection '{collection_name}' created/retrieved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        컬렉션 삭제
        
        Args:
            collection_name: 삭제할 컬렉션 이름
            
        Returns:
            bool: 성공 여부
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def add_vectors(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        documents: List[str]
    ) -> bool:
        """
        벡터 추가 (문서 임베딩 저장)
        
        Args:
            collection_name: 대상 컬렉션 이름
            ids: 문서 ID 리스트
            embeddings: 임베딩 벡터 리스트 (각 벡터는 float 리스트)
            metadatas: 메타데이터 리스트 (각 문서의 메타정보)
            documents: 원본 텍스트 리스트
            
        Returns:
            bool: 성공 여부
            
        예시:
        ```python
        store.add_vectors(
            collection_name="agent-001",
            ids=["doc1", "doc2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"topic": "python"}, {"topic": "java"}],
            documents=["Python is great", "Java is awesome"]
        )
        ```
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # ChromaDB에 데이터 추가
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(
                f"Added {len(ids)} vectors to collection '{collection_name}'"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to add vectors to collection '{collection_name}': {e}"
            )
            return False

    def search_vectors(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        유사도 기반 벡터 검색 (RAG의 핵심 기능!)
        
        Args:
            collection_name: 검색할 컬렉션 이름
            query_embedding: 질의 벡터 (사용자 질문의 임베딩)
            limit: 반환할 최대 결과 수
            where: 메타데이터 필터 조건 (예: {"category": "python"})
            
        Returns:
            검색 결과 리스트, 각 결과는 다음 형태:
            {
                'id': 문서 ID,
                'similarity_score': 유사도 점수 (0~1),
                'content': 원본 텍스트,
                'metadata': 메타데이터
            }
            
        예시:
        ```python
        results = store.search_vectors(
            collection_name="agent-001",
            query_embedding=[0.1, 0.2, ...],
            limit=5,
            where={"category": "python"}
        )
        
        for result in results:
            print(f"Score: {result['similarity_score']}")
            print(f"Content: {result['content']}")
        ```
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # ChromaDB 쿼리 파라미터 구성
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": limit
            }
            
            # 메타데이터 필터 추가 (있는 경우)
            if where:
                query_params["where"] = where
            
            # 검색 실행
            results = collection.query(**query_params)
            
            # 결과를 표준 형식으로 변환
            formatted_results = []
            
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'similarity_score': float(
                            1.0 - results['distances'][0][i]
                        ),  # ChromaDB는 distance 반환, 유사도로 변환
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            logger.info(
                f"Found {len(formatted_results)} results in "
                f"collection '{collection_name}'"
            )
            return formatted_results
            
        except Exception as e:
            logger.error(
                f"Failed to search vectors in collection '{collection_name}': {e}"
            )
            return []

    def get_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        특정 ID로 벡터 조회
        
        Args:
            collection_name: 컬렉션 이름
            ids: 조회할 문서 ID 리스트
            
        Returns:
            조회 결과 리스트
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # ID로 직접 조회
            results = collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )
            
            # 결과를 표준 형식으로 변환
            formatted_results = []
            
            if results['ids']:
                for i in range(len(results['ids'])):
                    formatted_results.append({
                        'id': results['ids'][i],
                        'similarity_score': 1.0,  # 직접 조회는 완전 일치
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    })
            
            logger.info(
                f"Retrieved {len(formatted_results)} vectors from "
                f"collection '{collection_name}'"
            )
            return formatted_results
            
        except Exception as e:
            logger.error(
                f"Failed to get vectors from collection '{collection_name}': {e}"
            )
            return []

    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        특정 벡터 삭제
        
        Args:
            collection_name: 컬렉션 이름
            ids: 삭제할 문서 ID 리스트
            
        Returns:
            bool: 성공 여부
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=ids)
            
            logger.info(
                f"Deleted {len(ids)} vectors from collection '{collection_name}'"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to delete vectors from collection '{collection_name}': {e}"
            )
            return False

    def count_vectors(self, collection_name: str) -> int:
        """
        컬렉션의 벡터 개수 조회
        
        Args:
            collection_name: 컬렉션 이름
            
        Returns:
            int: 벡터 개수
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            
            logger.debug(
                f"Collection '{collection_name}' has {count} vectors"
            )
            return count
            
        except Exception as e:
            logger.error(
                f"Failed to count vectors in collection '{collection_name}': {e}"
            )
            return 0

    def list_collections(self) -> List[str]:
        """
        모든 컬렉션 목록 조회
        
        Returns:
            List[str]: 컬렉션 이름 리스트
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            logger.info(f"Found {len(collection_names)} collections")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def health_check(self) -> bool:
        """
        ChromaDB 서버 연결 상태 확인
        
        Returns:
            bool: 연결 정상 여부
        """
        try:
            # heartbeat으로 연결 확인
            self.client.heartbeat()
            logger.info("ChromaDB health check: OK")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False
