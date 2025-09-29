#!/usr/bin/env python3
"""
ChromaDB 데이터 조회 도구

저장된 ChromaDB 데이터를 조회하고 검색하는 유틸리티입니다.

사용법:
    # 모든 컬렉션 목록 보기
    python tools/chroma_query.py --list
    
    # 특정 컬렉션의 모든 문서 보기
    python tools/chroma_query.py --collection demo_python_agent --show-all
    
    # 특정 컬렉션에서 검색
    python tools/chroma_query.py --collection demo_python_agent --search "Python 역사"
    
    # 컬렉션 상세 정보
    python tools/chroma_query.py --collection demo_python_agent --info
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from rag.vector_store import ChromaVectorStore
from typing import List
import numpy as np


class ChromaQueryTool:
    """ChromaDB 조회 도구"""
    
    def __init__(self, persist_directory: str = "./chroma-data"):
        """
        Args:
            persist_directory: ChromaDB 데이터 디렉토리
        """
        self.store = ChromaVectorStore(
            persist_directory=persist_directory,
            use_remote=False
        )
        print(f"📁 ChromaDB 연결: {persist_directory}")
        print()
    
    def list_collections(self):
        """모든 컬렉션 목록 표시"""
        collections = self.store.list_collections()
        
        print("=" * 70)
        print("📚 저장된 컬렉션 목록")
        print("=" * 70)
        
        if not collections:
            print("⚠️  저장된 컬렉션이 없습니다.")
            return
        
        for i, col_name in enumerate(collections, 1):
            count = self.store.count_vectors(col_name)
            print(f"{i}. {col_name}")
            print(f"   문서 수: {count}개")
            print()
    
    def collection_info(self, collection_name: str):
        """컬렉션 상세 정보 표시"""
        print("=" * 70)
        print(f"📊 컬렉션 정보: {collection_name}")
        print("=" * 70)
        
        try:
            count = self.store.count_vectors(collection_name)
            print(f"총 문서 수: {count}개")
            print()
            
            # 샘플 문서 몇 개 가져오기
            if count > 0:
                print("📄 샘플 문서 (최대 3개):")
                print("-" * 70)
                
                # ChromaDB에서 직접 가져오기
                collection = self.store.client.get_collection(name=collection_name)
                results = collection.get(
                    limit=3,
                    include=["documents", "metadatas"]
                )
                
                for i, (doc_id, doc, metadata) in enumerate(
                    zip(results['ids'], results['documents'], results['metadatas']), 1
                ):
                    print(f"\n[{i}] ID: {doc_id}")
                    print(f"    내용: {doc[:100]}{'...' if len(doc) > 100 else ''}")
                    if metadata:
                        print(f"    메타: {metadata}")
        
        except Exception as e:
            print(f"❌ 에러: {e}")
    
    def show_all_documents(self, collection_name: str, limit: int = None):
        """컬렉션의 모든 문서 표시"""
        print("=" * 70)
        print(f"📄 컬렉션의 모든 문서: {collection_name}")
        print("=" * 70)
        
        try:
            total_count = self.store.count_vectors(collection_name)
            print(f"총 문서 수: {total_count}개")
            
            if limit:
                print(f"표시 제한: {limit}개")
            
            print()
            
            # 모든 문서 가져오기
            collection = self.store.client.get_collection(name=collection_name)
            results = collection.get(
                limit=limit or total_count,
                include=["documents", "metadatas"]
            )
            
            for i, (doc_id, doc, metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas']), 1
            ):
                print(f"[{i}] ID: {doc_id}")
                print(f"    내용: {doc}")
                if metadata:
                    print(f"    메타: {metadata}")
                print()
        
        except Exception as e:
            print(f"❌ 에러: {e}")
    
    def search(self, collection_name: str, query: str, limit: int = 5):
        """텍스트 검색"""
        print("=" * 70)
        print(f"🔍 검색: {query}")
        print(f"   컬렉션: {collection_name}")
        print("=" * 70)
        print()
        
        try:
            # 간단한 임베딩 생성 (실제로는 동일한 임베딩 모델 사용해야 함)
            # 여기서는 텍스트를 직접 검색
            collection = self.store.client.get_collection(name=collection_name)
            
            # ChromaDB의 query 메서드 사용
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['ids'][0]:
                print("❌ 검색 결과가 없습니다.")
                return
            
            print(f"📚 검색 결과 ({len(results['ids'][0])}개):")
            print("-" * 70)
            
            for i, (doc_id, doc, metadata, distance) in enumerate(
                zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1
            ):
                similarity = 1.0 - distance  # distance를 similarity로 변환
                print(f"\n[{i}] 유사도: {similarity:.4f}")
                print(f"    ID: {doc_id}")
                print(f"    내용: {doc}")
                if metadata:
                    print(f"    메타: {metadata}")
        
        except Exception as e:
            print(f"❌ 에러: {e}")
    
    def filter_search(self, collection_name: str, where: dict, limit: int = 10):
        """메타데이터 필터링 검색"""
        print("=" * 70)
        print(f"🔍 필터 검색")
        print(f"   컬렉션: {collection_name}")
        print(f"   조건: {where}")
        print("=" * 70)
        print()
        
        try:
            collection = self.store.client.get_collection(name=collection_name)
            results = collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            if not results['ids']:
                print("❌ 검색 결과가 없습니다.")
                return
            
            print(f"📚 검색 결과 ({len(results['ids'])}개):")
            print("-" * 70)
            
            for i, (doc_id, doc, metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas']), 1
            ):
                print(f"\n[{i}] ID: {doc_id}")
                print(f"    내용: {doc}")
                if metadata:
                    print(f"    메타: {metadata}")
        
        except Exception as e:
            print(f"❌ 에러: {e}")
    
    def delete_collection(self, collection_name: str, confirm: bool = False):
        """컬렉션 삭제"""
        if not confirm:
            response = input(f"⚠️  정말로 '{collection_name}' 컬렉션을 삭제하시겠습니까? (yes/no): ")
            if response.lower() != 'yes':
                print("취소되었습니다.")
                return
        
        try:
            result = self.store.delete_collection(collection_name)
            if result:
                print(f"✅ 컬렉션 '{collection_name}'이 삭제되었습니다.")
            else:
                print(f"❌ 컬렉션 '{collection_name}'을 찾을 수 없습니다.")
        except Exception as e:
            print(f"❌ 에러: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB 데이터 조회 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모든 컬렉션 목록
  python tools/chroma_query.py --list
  
  # 컬렉션 정보
  python tools/chroma_query.py --collection demo_python_agent --info
  
  # 모든 문서 보기
  python tools/chroma_query.py --collection demo_python_agent --show-all
  
  # 텍스트 검색
  python tools/chroma_query.py --collection demo_python_agent --search "Python"
  
  # 필터 검색 (JSON 형식)
  python tools/chroma_query.py --collection demo_python_agent --filter '{"topic": "history"}'
  
  # 컬렉션 삭제
  python tools/chroma_query.py --collection test_collection --delete
        """
    )
    
    parser.add_argument(
        '--dir',
        default='./chroma-data',
        help='ChromaDB 데이터 디렉토리 (기본: ./chroma-data)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='모든 컬렉션 목록 표시'
    )
    
    parser.add_argument(
        '--collection',
        help='대상 컬렉션 이름'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='컬렉션 정보 표시'
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='컬렉션의 모든 문서 표시'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='표시할 문서 수 제한'
    )
    
    parser.add_argument(
        '--search',
        help='텍스트 검색'
    )
    
    parser.add_argument(
        '--filter',
        help='메타데이터 필터 검색 (JSON 형식)'
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help='컬렉션 삭제'
    )
    
    args = parser.parse_args()
    
    # ChromaDB 도구 초기화
    tool = ChromaQueryTool(persist_directory=args.dir)
    
    # 명령 실행
    if args.list:
        tool.list_collections()
    
    elif args.collection:
        if args.info:
            tool.collection_info(args.collection)
        
        elif args.show_all:
            tool.show_all_documents(args.collection, limit=args.limit)
        
        elif args.search:
            tool.search(args.collection, args.search, limit=args.limit or 5)
        
        elif args.filter:
            import json
            where_dict = json.loads(args.filter)
            tool.filter_search(args.collection, where_dict, limit=args.limit or 10)
        
        elif args.delete:
            tool.delete_collection(args.collection)
        
        else:
            # 기본: 컬렉션 정보 표시
            tool.collection_info(args.collection)
    
    else:
        # 인자가 없으면 도움말 표시
        parser.print_help()


if __name__ == "__main__":
    main()
