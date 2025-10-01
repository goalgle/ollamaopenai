#!/usr/bin/env python3
"""
파일 기반 문서 임포트 유틸리티

파일 포맷:
documents = [
  {
    "id": "doc_001",  # Optional: 생략 시 자동 생성
    "document": "문서 내용...",
    "metadata": {
      "key": "value"
    }
  },
  ...
]
"""

import sys
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class DocumentImporter:
    """문서 임포트 유틸리티"""
    
    @staticmethod
    def load_documents_from_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Python 파일에서 documents 변수를 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            문서 리스트
            
        Raises:
            FileNotFoundError: 파일이 없음
            ValueError: 파일 형식 오류
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Python 코드 실행하여 documents 변수 추출
        local_vars = {}
        try:
            exec(content, {}, local_vars)
        except Exception as e:
            raise ValueError(f"Failed to parse file: {e}")
        
        # documents 변수 확인
        if 'documents' not in local_vars:
            raise ValueError("File must contain 'documents' variable")
        
        documents = local_vars['documents']
        
        if not isinstance(documents, list):
            raise ValueError("'documents' must be a list")
        
        return documents
    
    @staticmethod
    def validate_document(doc: Dict[str, Any], index: int) -> None:
        """
        문서 형식 검증
        
        Args:
            doc: 문서 딕셔너리
            index: 문서 인덱스 (에러 메시지용)
            
        Raises:
            ValueError: 형식 오류
        """
        if not isinstance(doc, dict):
            raise ValueError(f"Document at index {index} must be a dictionary")
        
        # document 필드 필수
        if 'document' not in doc:
            raise ValueError(f"Document at index {index} missing 'document' field")
        
        # document는 문자열이어야 함
        if not isinstance(doc['document'], str):
            raise ValueError(f"Document at index {index}: 'document' must be a string")
        
        # metadata는 딕셔너리여야 함 (있는 경우)
        if 'metadata' in doc and not isinstance(doc['metadata'], dict):
            raise ValueError(f"Document at index {index}: 'metadata' must be a dictionary")
    
    @staticmethod
    def prepare_documents(
        documents: List[Dict[str, Any]], 
        auto_generate_id: bool = True,
        add_import_metadata: bool = True
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        ChromaDB 형식으로 문서 준비
        
        Args:
            documents: 원본 문서 리스트
            auto_generate_id: ID 자동 생성 여부 (ID가 없는 경우)
            add_import_metadata: 임포트 메타데이터 추가 여부
            
        Returns:
            (ids, documents, metadatas) 튜플
            
        Raises:
            ValueError: ID 중복 또는 형식 오류
        """
        ids = []
        docs = []
        metadatas = []
        
        seen_ids = set()
        
        for i, doc in enumerate(documents):
            # 검증
            DocumentImporter.validate_document(doc, i)
            
            # ID 처리
            doc_id = doc.get('id')
            
            if doc_id is None:
                if auto_generate_id:
                    # ID 자동 생성
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                else:
                    raise ValueError(f"Document at index {i} missing 'id' field (auto_generate_id=False)")
            
            # ID 타입 확인
            if not isinstance(doc_id, str):
                raise ValueError(f"Document at index {i}: 'id' must be a string")
            
            # ID 중복 확인
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            
            seen_ids.add(doc_id)
            ids.append(doc_id)
            
            # 문서 내용
            docs.append(doc['document'])
            
            # 메타데이터
            metadata = doc.get('metadata', {}).copy()
            
            # 임포트 메타데이터 추가
            if add_import_metadata:
                metadata['imported_at'] = str(datetime.now())
                metadata['import_source'] = 'file_import'
            
            metadatas.append(metadata)
        
        return ids, docs, metadatas
    
    @staticmethod
    def import_to_collection(
        collection,
        file_path: str,
        auto_generate_id: bool = True,
        batch_size: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        파일에서 문서를 읽어 콜렉션에 임포트
        
        Args:
            collection: ChromaDB 콜렉션
            file_path: 파일 경로
            auto_generate_id: ID 자동 생성 여부
            batch_size: 배치 크기
            verbose: 진행 상황 출력 여부
            
        Returns:
            임포트 결과 통계
        """
        if verbose:
            print(f"\n📂 Loading documents from: {file_path}")
        
        # 파일 로드
        try:
            documents = DocumentImporter.load_documents_from_file(file_path)
        except Exception as e:
            if verbose:
                print(f"❌ Error loading file: {e}")
            raise
        
        if verbose:
            print(f"✅ Loaded {len(documents)} documents from file")
        
        # 문서 준비
        try:
            ids, docs, metadatas = DocumentImporter.prepare_documents(
                documents, 
                auto_generate_id=auto_generate_id
            )
        except Exception as e:
            if verbose:
                print(f"❌ Error preparing documents: {e}")
            raise
        
        if verbose:
            print(f"✅ Prepared {len(ids)} documents for import")
            
            # ID 생성 통계
            auto_generated = sum(1 for doc in documents if doc.get('id') is None)
            if auto_generated > 0:
                print(f"   - Auto-generated IDs: {auto_generated}")
            if len(documents) - auto_generated > 0:
                print(f"   - Custom IDs: {len(documents) - auto_generated}")
        
        # 배치로 임포트
        total = len(ids)
        imported = 0
        errors = []
        
        if verbose:
            print(f"\n📥 Importing documents (batch_size={batch_size})...")
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_ids = ids[i:batch_end]
            batch_docs = docs[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadatas
                )
                imported += len(batch_ids)
                
                if verbose:
                    print(f"   Batch {i//batch_size + 1}: {len(batch_ids)} documents ✓")
                    
            except Exception as e:
                error_msg = f"Batch {i//batch_size + 1} failed: {e}"
                errors.append(error_msg)
                if verbose:
                    print(f"   Batch {i//batch_size + 1}: ✗ {e}")
        
        # 결과
        result = {
            'total': total,
            'imported': imported,
            'failed': total - imported,
            'errors': errors
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Import completed")
            print(f"   Total: {result['total']}")
            print(f"   Imported: {result['imported']}")
            if result['failed'] > 0:
                print(f"   Failed: {result['failed']}")
                for error in errors:
                    print(f"      - {error}")
            print(f"{'='*60}\n")
        
        return result
    
    @staticmethod
    def preview_file(file_path: str, max_docs: int = 5) -> None:
        """
        파일 내용 미리보기
        
        Args:
            file_path: 파일 경로
            max_docs: 최대 표시 문서 개수
        """
        print(f"\n📄 File Preview: {file_path}")
        print("="*60)
        
        try:
            documents = DocumentImporter.load_documents_from_file(file_path)
            
            print(f"Total documents: {len(documents)}\n")
            
            for i, doc in enumerate(documents[:max_docs]):
                print(f"Document {i+1}:")
                print(f"  ID: {doc.get('id', '(auto-generate)')}")
                
                content = doc.get('document', '')
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  Content: {content_preview}")
                
                metadata = doc.get('metadata', {})
                print(f"  Metadata: {metadata}")
                print()
            
            if len(documents) > max_docs:
                print(f"... and {len(documents) - max_docs} more documents")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """테스트 및 CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Document Import Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview file
  python tools/file_import.py --preview sample_documents.py
  
  # Validate file format
  python tools/file_import.py --validate sample_documents.py
        """
    )
    
    parser.add_argument(
        'file',
        help='Document file path'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview file contents'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate file format'
    )
    
    parser.add_argument(
        '--max-docs',
        type=int,
        default=5,
        help='Maximum documents to preview (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.preview:
        DocumentImporter.preview_file(args.file, args.max_docs)
    elif args.validate:
        try:
            documents = DocumentImporter.load_documents_from_file(args.file)
            ids, docs, metadatas = DocumentImporter.prepare_documents(documents)
            print(f"✅ File is valid")
            print(f"   Documents: {len(documents)}")
            print(f"   Auto-generated IDs: {sum(1 for doc in documents if doc.get('id') is None)}")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)
    else:
        # 기본: preview
        DocumentImporter.preview_file(args.file, args.max_docs)


if __name__ == "__main__":
    main()
