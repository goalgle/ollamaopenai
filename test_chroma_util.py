#!/usr/bin/env python3
"""
ChromaDB 유틸리티 테스트 프로그램
프롬프트 기반으로 ChromaDB를 쉽게 탐색할 수 있습니다.

사용 가능한 명령어:
    - collections : 모든 콜렉션 목록 출력
    - info <collection_name> : 콜렉션 정보 출력
    - show <collection_name> [start] [size] : 문서 출력
    - search <collection_name> <query> [limit] : 유사도 검색
    - filter <min_similarity> : 마지막 결과에서 유사도 필터링
    - health : ChromaDB 연결 상태 확인
    - help : 도움말
    - exit : 종료
"""

import sys
import os
import readline
import atexit
import uuid
from typing import Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.chroma_util import ChromaUtil, DocumentResults


class ChromaUtilCLI:
    """ChromaDB 유틸리티 CLI"""
    
    def __init__(self, persist_directory: str = "./chroma-data", use_remote: bool = False):
        self.chroma = ChromaUtil(
            persist_directory=persist_directory,
            use_remote=use_remote
        )
        self.last_results: Optional[DocumentResults] = None
        self.original_results: Optional[DocumentResults] = None  # 원본 결과 보관
        self.running = True
        self.history_file = os.path.expanduser("~/.chroma_util_history")
        
        # readline 설정
        self._setup_readline()
    
    def _setup_readline(self):
        """readline 설정 (히스토리 및 자동완성)"""
        # 히스토리 파일 로드
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
        
        # 히스토리 크기 설정
        readline.set_history_length(1000)
        
        # 프로그램 종료 시 히스토리 저장
        atexit.register(self._save_history)
        
        # 자동완성 설정
        readline.set_completer(self._completer)
        readline.parse_and_bind('tab: complete')
        
        # Mac에서는 libedit를 사용하므로 다른 설정 필요
        if 'libedit' in readline.__doc__:
            readline.parse_and_bind("bind ^I rl_complete")
    
    def _save_history(self):
        """히스토리 파일 저장"""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            pass  # 히스토리 저장 실패해도 프로그램은 계속
    
    def _completer(self, text: str, state: int):
        """자동완성 함수"""
        # 사용 가능한 명령어 목록
        commands = [
            'collections', 'info', 'show', 'search', 'filter', 
            'metadata', 'top', 'reset', 'create', 'add', 'delete', 'drop',
            'health', 'history', 'help', 'clear', 'exit', 'quit'
        ]
        
        # 콜렉션 이름도 자동완성에 추가
        try:
            collections = self.chroma.client.list_collections()
            collection_names = [col.name for col in collections]
        except:
            collection_names = []
        
        # text로 시작하는 명령어 필터링
        options = [cmd for cmd in commands if cmd.startswith(text)]
        
        # 공백이 있으면 콜렉션 이름 자동완성
        line = readline.get_line_buffer()
        if ' ' in line:
            options = [name for name in collection_names if name.startswith(text)]
        
        if state < len(options):
            return options[state]
        return None
    
    def show_welcome(self):
        """환영 메시지"""
        print("\n" + "="*70)
        print("  ChromaDB Utility - Interactive CLI")
        print("="*70)
        print("\n💡 Type 'help' to see available commands")
        print("💡 Use ↑↓ arrows to navigate command history")
        print("💡 Use TAB for auto-completion")
        print("💡 Filters work as AND conditions (stack on each other)")
        print("💡 Use 'reset' to clear all filters\n")
    
    def show_help(self, args: list):
        """도움말 출력"""
        help_text = """
╔═══════════════════════════════════════════════════════════════════╗
║                     Available Commands                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  VIEWING COMMANDS                                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  collections                                                       ║
║    → Show all collections                                         ║
║                                                                   ║
║  info <collection_name>                                           ║
║    → Show collection information                                  ║
║    → Example: info my_collection                                  ║
║    → TAB: Auto-complete collection names                          ║
║                                                                   ║
║  show <collection_name> [start] [size]                            ║
║    → Show documents in a collection                               ║
║    → Example: show my_collection 0 10                             ║
║    → Default: start=0, size=10                                    ║
║                                                                   ║
║  search <collection_name> <query> [limit]                         ║
║    → Search similar documents                                     ║
║    → Example: search my_collection "Python programming" 5         ║
║    → Default: limit=10                                            ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  FILTERING COMMANDS (AND conditions)                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  filter <min_similarity>                                          ║
║    → Filter current results by similarity (AND condition)         ║
║    → Example: filter 0.5   (similarity >= 0.5)                    ║
║    → Applies on top of previous filters                           ║
║                                                                   ║
║  metadata <key> <value>                                           ║
║    → Filter current results by metadata (AND condition)           ║
║    → Example: metadata category python                            ║
║    → Applies on top of previous filters                           ║
║                                                                   ║
║  top <count>                                                      ║
║    → Show top N documents from current results (AND condition)    ║
║    → Example: top 5    (show top 5 documents)                     ║
║    → Automatically sorts by similarity                            ║
║                                                                   ║
║  reset                                                            ║
║    → Clear all filters and return to original search results      ║
║    → Use this to start filtering from scratch                     ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  EDITING COMMANDS                                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  create <collection_name>                                         ║
║    → Create a new collection                                      ║
║    → Example: create my_new_collection                            ║
║                                                                   ║
║  add <collection_name> <content>                                  ║
║    → Add a document to a collection                               ║
║    → ID is auto-generated if not specified                        ║
║    → Examples:                                                    ║
║      add my_docs 'Python is great'                                ║
║      add my_docs 'AI tutorial' --id tutorial_001                  ║
║      add tech_docs 'Article' --meta category=tech author=John     ║
║    → Use quotes for multi-word content                            ║
║    → Options:                                                     ║
║        --id <doc_id>           : Specify custom ID                ║
║        --meta key=val key=val  : Add metadata                     ║
║                                                                   ║
║  delete <collection_name> <doc_id>                                ║
║    → Delete a document from a collection                          ║
║    → Example: delete my_collection doc_001                        ║
║                                                                   ║
║  drop <collection_name>                                           ║
║    → Delete entire collection (requires confirmation)             ║
║    → Example: drop my_collection                                  ║
║    → Warning: This deletes all documents!                         ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  UTILITY COMMANDS                                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  health                                                           ║
║    → Check ChromaDB connection status                             ║
║                                                                   ║
║  history                                                          ║
║    → Show recent command history                                  ║
║                                                                   ║
║  clear                                                            ║
║    → Clear screen                                                 ║
║                                                                   ║
║  help                                                             ║
║    → Show this help message                                       ║
║                                                                   ║
║  exit / quit                                                      ║
║    → Exit the program                                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Keyboard Shortcuts                                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  ↑ / ↓        : Navigate command history                          ║
║  TAB          : Auto-complete commands and collection names       ║
║  Ctrl+C       : Cancel current input                              ║
║  Ctrl+D       : Exit (same as 'exit')                             ║
╚═══════════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def handle_collections(self, args: list):
        """콜렉션 목록 출력"""
        self.chroma.show_collections()
    
    def handle_info(self, args: list):
        """콜렉션 정보 출력"""
        if len(args) < 1:
            print("❌ Error: Collection name required")
            print("Usage: info <collection_name>")
            return
        
        collection_name = args[0]
        self.chroma.get_collection_info(collection_name)
    
    def handle_show(self, args: list):
        """문서 출력"""
        if len(args) < 1:
            print("❌ Error: Collection name required")
            print("Usage: show <collection_name> [start] [size]")
            return
        
        collection_name = args[0]
        start = int(args[1]) if len(args) > 1 else 0
        size = int(args[2]) if len(args) > 2 else 10
        
        results = self.chroma.show_documents(
            collection_name, start, size
        )
        
        # 새로운 검색이므로 원본과 현재 결과 모두 업데이트
        self.original_results = results
        self.last_results = results
    
    def handle_search(self, args: list):
        """유사도 검색"""
        if len(args) < 2:
            print("❌ Error: Collection name and query required")
            print("Usage: search <collection_name> <query> [limit]")
            return
        
        collection_name = args[0]
        # 쿼리는 여러 단어일 수 있으므로 마지막 인자가 숫자가 아니면 전부 합침
        limit = 10
        query_parts = args[1:]
        
        # 마지막 인자가 숫자면 limit로 사용
        if query_parts[-1].isdigit():
            limit = int(query_parts[-1])
            query_parts = query_parts[:-1]
        
        query = " ".join(query_parts)
        
        print(f"🔍 Searching for: '{query}'")
        results = self.chroma.search_similar(
            collection_name, query, limit
        )
        
        # 새로운 검색이므로 원본과 현재 결과 모두 업데이트
        self.original_results = results
        self.last_results = results
    
    def handle_filter(self, args: list):
        """마지막 결과 필터링 (AND 조건)"""
        if self.last_results is None:
            print("❌ Error: No previous search results")
            print("Run a search first using 'search' or 'show' command")
            return
        
        if len(args) < 1:
            print("❌ Error: Minimum similarity required")
            print("Usage: filter <min_similarity>")
            print("\nExamples:")
            print("  filter 0.5    # Show documents with similarity >= 0.5")
            print("  filter 0      # Show all documents (no filter)")
            print("  filter -1     # Show documents with similarity >= -1")
            print("\n💡 Tip: Filters are applied as AND conditions")
            print("        Use 'reset' to start over from search results")
            return
        
        try:
            min_similarity = float(args[0])
            
            print(f"\n🔎 Filtering current results (similarity >= {min_similarity})")
            print(f"Before: {len(self.last_results)} documents")
            
            # 현재 유사도 범위 표시
            if len(self.last_results) > 0:
                scores = [d.similarity_score for d in self.last_results]
                print(f"Similarity range: {min(scores):.4f} ~ {max(scores):.4f}")
            
            # 현재 결과에서 필터링 (AND 조건)
            self.last_results = self.last_results.get_similarity_gte(min_similarity)
            
            print(f"After filter: {len(self.last_results)} documents")
            print(f"💡 Use 'reset' to go back to original search results\n")
            
            if len(self.last_results) == 0:
                print("⚠️  No documents match the filter criteria")
                print(f"   Try 'reset' and use a lower threshold")
            else:
                print(self.last_results)
            
        except ValueError:
            print("❌ Error: Similarity must be a number")
            print("Examples: filter 0.5, filter 0, filter -0.5")
    
    def handle_metadata(self, args: list):
        """메타데이터로 필터링 (AND 조건)"""
        if self.last_results is None:
            print("❌ Error: No previous search results")
            return
        
        if len(args) < 2:
            print("❌ Error: Key and value required")
            print("Usage: metadata <key> <value>")
            return
        
        key = args[0]
        value = args[1]
        
        print(f"\n🔎 Filtering current results by metadata: {key}={value}")
        print(f"Before: {len(self.last_results)} documents")
        
        # 현재 결과에서 필터링 (AND 조건)
        self.last_results = self.last_results.filter_by_metadata(key, value)
        
        print(f"After filter: {len(self.last_results)} documents")
        print(f"💡 Use 'reset' to go back to original search results\n")
        
        if len(self.last_results) == 0:
            print("⚠️  No documents match the metadata filter")
            print(f"   Try 'reset' and use different criteria")
        else:
            print(self.last_results)
    
    def handle_health(self, args: list):
        """헬스 체크"""
        self.chroma.health_check()
    
    def handle_create(self, args: list):
        """새 콜렉션 생성"""
        if len(args) < 1:
            print("❌ Error: Collection name required")
            print("Usage: create <collection_name>")
            print("\nExample:")
            print("  create my_collection")
            return
        
        collection_name = args[0]
        
        try:
            # 이미 존재하는지 확인
            collections = self.chroma.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if collection_name in collection_names:
                print(f"⚠️  Collection '{collection_name}' already exists")
                return
            
            # 생성
            self.chroma.create_collection(collection_name)
            print(f"✅ Collection '{collection_name}' created successfully")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def handle_add(self, args: list):
        """콜렉션에 문서 추가"""
        if len(args) < 2:
            print("❌ Error: Collection name and content required")
            print("\nUsage:")
            print("  add <collection_name> <content>")
            print("  add <collection_name> <content> --id <doc_id>")
            print("  add <collection_name> <content> --meta key1=value1 key2=value2")
            print("\nExamples:")
            print("  add my_docs 'Python is great'")
            print("    → Auto-generated ID: doc_abc123")
            print()
            print("  add my_docs 'Python tutorial' --id tutorial_001")
            print("    → Custom ID: tutorial_001")
            print()
            print("  add tech_docs 'AI article' --meta category=tech author=John year=2024")
            print("    → With metadata")
            print()
            print("💡 Tip: ID is auto-generated if not specified")
            print("💡 Tip: Use quotes for multi-word content")
            return
        
        collection_name = args[0]
        
        # 인자 파싱
        doc_id = None
        metadata = {"added_by": "cli", "timestamp": str(datetime.now())}
        content_parts = []
        
        i = 1
        while i < len(args):
            if args[i] == '--id' and i + 1 < len(args):
                doc_id = args[i + 1]
                i += 2
            elif args[i] == '--meta':
                # --meta 다음부터 key=value 형식 파싱
                i += 1
                while i < len(args) and '=' in args[i]:
                    key, value = args[i].split('=', 1)
                    # 타입 추론 (숫자면 int로 변환)
                    if value.isdigit():
                        metadata[key] = int(value)
                    elif value.lower() in ['true', 'false']:
                        metadata[key] = value.lower() == 'true'
                    else:
                        metadata[key] = value
                    i += 1
            else:
                content_parts.append(args[i])
                i += 1
        
        content = " ".join(content_parts)
        
        if not content:
            print("❌ Error: Content cannot be empty")
            return
        
        # doc_id 자동 생성
        if not doc_id:
            import uuid
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        try:
            collection = self.chroma.client.get_collection(name=collection_name)
            
            # 문서 추가
            collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata]
            )
            
            print(f"\n✅ Document added successfully")
            print(f"   Collection: {collection_name}")
            print(f"   ID: {doc_id}")
            print(f"   Metadata: {metadata}")
            print(f"   Content: {content[:100]}{'...' if len(content) > 100 else ''}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if "does not exist" in str(e) or "not exist" in str(e):
                print(f"   Collection '{collection_name}' does not exist")
                print(f"   Create it first: create {collection_name}")
    
    def handle_delete_doc(self, args: list):
        """콜렉션에서 문서 삭제"""
        if len(args) < 2:
            print("❌ Error: Collection name and document ID required")
            print("Usage: delete <collection_name> <doc_id>")
            print("\nExample:")
            print("  delete my_collection doc_001")
            return
        
        collection_name = args[0]
        doc_id = args[1]
        
        try:
            collection = self.chroma.client.get_collection(name=collection_name)
            collection.delete(ids=[doc_id])
            
            print(f"✅ Document '{doc_id}' deleted from '{collection_name}'")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def handle_drop(self, args: list):
        """콜렉션 삭제"""
        if len(args) < 1:
            print("❌ Error: Collection name required")
            print("Usage: drop <collection_name>")
            print("\n⚠️  Warning: This will delete all documents in the collection!")
            return
        
        collection_name = args[0]
        
        # 확인
        confirm = input(f"⚠️  Are you sure you want to delete '{collection_name}'? (yes/no): ")
        
        if confirm.lower() != 'yes':
            print("❌ Cancelled")
            return
        
        self.chroma.delete_collection(collection_name)
    
    def handle_reset(self, args: list):
        """필터를 초기화하고 원본 검색 결과로 되돌림"""
        if self.original_results is None:
            print("❌ Error: No search results to reset")
            return
        
        print(f"\n🔄 Resetting to original search results")
        print(f"Original: {len(self.original_results)} documents")
        
        self.last_results = self.original_results
        
        print(f"✅ Filter reset complete\n")
        print(self.last_results)
    
    def handle_top(self, args: list):
        """유사도 높은 순서로 상위 N개만 표시 (AND 조건)"""
        if self.last_results is None:
            print("❌ Error: No search results")
            print("Run a search first using 'search' or 'show' command")
            return
        
        if len(args) < 1:
            print("❌ Error: Number of documents required")
            print("Usage: top <count>")
            print("\nExamples:")
            print("  top 5     # Show top 5 documents by similarity")
            print("  top 10    # Show top 10 documents by similarity")
            return
        
        try:
            count = int(args[0])
            
            if count <= 0:
                print("❌ Error: Count must be positive")
                return
            
            print(f"\n🏆 Top {count} documents by similarity")
            print(f"Before: {len(self.last_results)} documents")
            
            # 현재 결과를 유사도로 정렬 후 상위 N개 (AND 조건)
            self.last_results = (
                self.last_results
                .sort_by_similarity(reverse=True)
                .limit(count)
            )
            
            print(f"After top: {len(self.last_results)} documents")
            print(f"💡 Use 'reset' to go back to original search results\n")
            
            if len(self.last_results) == 0:
                print("⚠️  No documents available")
            else:
                # 순위 표시
                for i, doc in enumerate(self.last_results, 1):
                    print(f"🥇 Rank {i}")
                    print(f"   ID: {doc.id}")
                    print(f"   Similarity: {doc.similarity_score:.4f}")
                    print(f"   Metadata: {doc.metadata}")
                    content_preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                    print(f"   Content: {content_preview}")
                    print()
            
        except ValueError:
            print("❌ Error: Count must be a number")
            print("Example: top 5")
    
    def handle_clear(self, args: list):
        """화면 클리어"""
        os.system('clear' if os.name != 'nt' else 'cls')
        self.show_welcome()
    
    def handle_history(self, args: list):
        """명령어 히스토리 출력"""
        print("\n" + "="*60)
        print("Command History (last 20)")
        print("="*60)
        
        history_len = readline.get_current_history_length()
        start = max(1, history_len - 19)
        
        for i in range(start, history_len + 1):
            try:
                cmd = readline.get_history_item(i)
                if cmd:
                    print(f"  {i:3d}  {cmd}")
            except:
                pass
        
        print("="*60)
        print(f"Total: {history_len} commands")
        print("Use ↑↓ arrows to navigate history\n")
    
    def handle_exit(self, args: list):
        """종료"""
        print("\n👋 Goodbye!\n")
        self.running = False
    
    def process_command(self, command: str):
        """명령어 처리"""
        if not command.strip():
            return
        
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        # 명령어 매핑
        commands = {
            'collections': self.handle_collections,
            'info': self.handle_info,
            'show': self.handle_show,
            'search': self.handle_search,
            'filter': self.handle_filter,
            'metadata': self.handle_metadata,
            'top': self.handle_top,
            'reset': self.handle_reset,
            'create': self.handle_create,
            'add': self.handle_add,
            'delete': self.handle_delete_doc,
            'drop': self.handle_drop,
            'health': self.handle_health,
            'history': self.handle_history,
            'help': self.show_help,
            'clear': self.handle_clear,
            'exit': self.handle_exit,
            'quit': self.handle_exit,
        }
        
        if cmd in commands:
            try:
                commands[cmd](args)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type 'help' to see available commands")
    
    def run(self):
        """메인 루프"""
        self.show_welcome()
        
        while self.running:
            try:
                command = input("chroma> ").strip()
                if command:
                    self.process_command(command)
                    print()  # 빈 줄 추가
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit")
            except EOFError:
                break


def main():
    """메인 함수"""
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="ChromaDB Utility - Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_chroma_util.py
  python test_chroma_util.py ./my-chroma-data
  python test_chroma_util.py --dir ./stock-rag-data
  python test_chroma_util.py ./chroma-data --remote
        """
    )
    
    # 위치 인자로 디렉토리 받기
    parser.add_argument(
        'directory',
        nargs='?',  # optional positional argument
        default='./chroma-data',
        help='ChromaDB persist directory (default: ./chroma-data)'
    )
    
    # 또는 --dir 옵션으로도 받기 (위치 인자보다 우선)
    parser.add_argument(
        '--dir',
        dest='dir_option',
        help='ChromaDB persist directory (alternative to positional arg)'
    )
    
    parser.add_argument(
        '--remote',
        action='store_true',
        help='Use remote ChromaDB server instead of local'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Remote ChromaDB host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Remote ChromaDB port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # --dir 옵션이 있으면 그것을 사용, 없으면 위치 인자 사용
    persist_dir = args.dir_option if args.dir_option else args.directory
    
    # 시작 메시지
    print("\n" + "="*70)
    print("  ChromaDB Utility - Interactive CLI")
    print("="*70)
    print(f"\n📁 Database Directory: {persist_dir}")
    
    # 디렉토리 존재 확인 (로컬 모드일 때만)
    if not args.remote:
        if not os.path.exists(persist_dir):
            print(f"📂 Directory does not exist. Creating: {persist_dir}")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory created successfully")
        else:
            print(f"✅ Directory exists")
        print(f"💾 Local Mode")
    else:
        print(f"🌐 Remote Mode: {args.host}:{args.port}")
    
    print()
    
    cli = ChromaUtilCLI(
        persist_directory=persist_dir,
        use_remote=args.remote
    )
    
    # DB 연결 확인
    print("🔌 Connecting to ChromaDB...")
    if cli.chroma.health_check():
        print()
    else:
        print("⚠️  Warning: ChromaDB connection failed")
        print("   Continuing anyway... some commands may not work\n")
    
    cli.run()


if __name__ == "__main__":
    main()
