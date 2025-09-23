#!/usr/bin/env python3
"""
RAG Performance and Load Testing Suite
Comprehensive performance testing for RAG system and Vector Database operations
"""

import sys
from pathlib import Path

# Add project root to sys.path to find rag module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import time
import statistics
import psutil
import threading
import concurrent.futures
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

# Performance testing utilities
@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    operation_name: str
    total_operations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    concurrent_users: int = 10
    operations_per_user: int = 100
    ramp_up_time: float = 5.0
    test_duration: float = 60.0
    think_time: float = 0.1


class PerformanceProfiler:
    """Utility for measuring performance metrics"""

    def __init__(self):
        self.measurements = []
        self.start_memory = None
        self.start_cpu = None

    def start_profiling(self):
        """Start system resource monitoring"""
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.start_cpu = psutil.cpu_percent(interval=None)

    def measure_operation(self, operation_name: str):
        """Context manager for measuring individual operations"""
        return OperationTimer(self, operation_name)

    def get_metrics(self, operation_name: str) -> PerformanceMetrics:
        """Calculate performance metrics for an operation"""
        operation_times = [m['duration'] for m in self.measurements if m['operation'] == operation_name]

        if not operation_times:
            raise ValueError(f"No measurements found for operation: {operation_name}")

        # Calculate statistics
        total_time = sum(operation_times)
        avg_time = statistics.mean(operation_times)
        min_time = min(operation_times)
        max_time = max(operation_times)
        median_time = statistics.median(operation_times)

        # Percentiles
        sorted_times = sorted(operation_times)
        p95_time = sorted_times[int(len(sorted_times) * 0.95)]
        p99_time = sorted_times[int(len(sorted_times) * 0.99)]

        # Throughput
        throughput = len(operation_times) / total_time if total_time > 0 else 0

        # Resource usage
        current_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_usage = current_memory - (self.start_memory or current_memory)
        cpu_usage = psutil.cpu_percent(interval=None)

        return PerformanceMetrics(
            operation_name=operation_name,
            total_operations=len(operation_times),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )

    def export_metrics(self, filepath: str):
        """Export all measurements to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.measurements, f, indent=2, default=str)


class OperationTimer:
    """Context manager for timing operations"""

    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.profiler.measurements.append({
            'operation': self.operation_name,
            'duration': duration,
            'timestamp': time.time(),
            'success': exc_type is None
        })


class RAGLoadTester:
    """Load testing framework for RAG system"""

    def __init__(self, knowledge_manager, agent_id: str):
        self.knowledge_manager = knowledge_manager
        self.agent_id = agent_id
        self.results = []

    def run_concurrent_operations(
        self,
        operation_func,
        config: LoadTestConfig,
        operation_args: List[Any] = None
    ) -> List[Dict[str, Any]]:
        """Run operations concurrently with specified load pattern"""

        def user_simulation(user_id: int, operations_per_user: int, args_list: List[Any]):
            """Simulate single user operations"""
            user_results = []
            for i in range(operations_per_user):
                start_time = time.time()
                try:
                    args = args_list[i % len(args_list)] if args_list else []
                    result = operation_func(*args) if args else operation_func()
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)

                duration = time.time() - start_time
                user_results.append({
                    'user_id': user_id,
                    'operation_index': i,
                    'duration': duration,
                    'success': success,
                    'error': error,
                    'timestamp': time.time()
                })

                # Think time between operations
                if config.think_time > 0:
                    time.sleep(config.think_time)

            return user_results

        # Prepare operation arguments
        if operation_args is None:
            operation_args = [[] for _ in range(config.operations_per_user)]

        # Start concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            # Submit user simulations
            futures = []
            for user_id in range(config.concurrent_users):
                future = executor.submit(
                    user_simulation,
                    user_id,
                    config.operations_per_user,
                    operation_args
                )
                futures.append(future)

                # Ramp up delay
                if config.ramp_up_time > 0:
                    time.sleep(config.ramp_up_time / config.concurrent_users)

            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                user_results = future.result()
                all_results.extend(user_results)

        return all_results

    def analyze_load_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze load test results and generate summary"""
        if not results:
            return {"error": "No results to analyze"}

        # Basic statistics
        total_operations = len(results)
        successful_operations = sum(1 for r in results if r['success'])
        failed_operations = total_operations - successful_operations
        success_rate = (successful_operations / total_operations) * 100

        # Duration statistics (successful operations only)
        successful_durations = [r['duration'] for r in results if r['success']]
        if successful_durations:
            avg_duration = statistics.mean(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            median_duration = statistics.median(successful_durations)
            p95_duration = np.percentile(successful_durations, 95)
            p99_duration = np.percentile(successful_durations, 99)
        else:
            avg_duration = min_duration = max_duration = median_duration = p95_duration = p99_duration = 0

        # Throughput calculation
        start_time = min(r['timestamp'] - r['duration'] for r in results)
        end_time = max(r['timestamp'] for r in results)
        total_time = end_time - start_time
        throughput = successful_operations / total_time if total_time > 0 else 0

        # Error analysis
        error_counts = {}
        for result in results:
            if not result['success'] and result['error']:
                error_type = type(Exception(result['error'])).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            'summary': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'success_rate_percent': success_rate,
                'total_duration_seconds': total_time,
                'throughput_ops_per_sec': throughput
            },
            'timing_stats': {
                'avg_duration': avg_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'median_duration': median_duration,
                'p95_duration': p95_duration,
                'p99_duration': p99_duration
            },
            'errors': error_counts
        }


# Test fixtures for performance testing
@pytest.fixture
def performance_rag_system(tmp_path):
    """Setup RAG system optimized for performance testing"""
    from test.rag_test_implementations import MockVectorStore, MockEmbeddingService
    from rag.knowledge_manager import KnowledgeManager

    # Use faster mock implementations for performance testing
    vector_store = MockVectorStore()
    embedding_service = MockEmbeddingService(dimension=768)
    knowledge_manager = KnowledgeManager(
        vector_store,
        embedding_service,
        str(tmp_path / "perf_metadata.db")
    )

    # Create test agent
    agent_id = "perf-test-agent"
    knowledge_manager.create_agent_collection(agent_id, "Performance Test Agent", "test")

    return knowledge_manager, agent_id


@pytest.fixture
def performance_profiler():
    """Provide performance profiler instance"""
    profiler = PerformanceProfiler()
    profiler.start_profiling()
    return profiler


# Performance Tests
class TestRAGPerformance:
    """Performance tests for RAG system components"""

    @pytest.mark.performance
    def test_knowledge_storage_performance(self, performance_rag_system, performance_profiler):
        """Test performance of knowledge storage operations"""
        km, agent_id = performance_rag_system

        # Test different document sizes
        test_cases = [
            ("small", 100, 50),    # 50 small documents
            ("medium", 1000, 20),   # 20 medium documents
            ("large", 5000, 10),    # 10 large documents
        ]

        for size_name, content_size, num_docs in test_cases:
            operation_name = f"store_knowledge_{size_name}"

            for i in range(num_docs):
                content = f"Test content " * (content_size // 13)  # Approximate size

                with performance_profiler.measure_operation(operation_name):
                    knowledge_id = km.store_knowledge(
                        agent_id=agent_id,
                        content=content,
                        metadata={"size": size_name, "index": i},
                        tags=[size_name, "performance_test"],
                        source="perf_test"
                    )
                    assert knowledge_id is not None

            # Analyze metrics for this size category
            metrics = performance_profiler.get_metrics(operation_name)

            # Performance assertions
            assert metrics.avg_time < 1.0, f"Average storage time too high for {size_name}: {metrics.avg_time}s"
            assert metrics.p95_time < 2.0, f"95th percentile too high for {size_name}: {metrics.p95_time}s"
            assert metrics.throughput_ops_per_sec > 1.0, f"Throughput too low for {size_name}: {metrics.throughput_ops_per_sec} ops/s"

            print(f"\n{size_name.upper()} DOCUMENTS STORAGE PERFORMANCE:")
            print(f"  Average time: {metrics.avg_time:.3f}s")
            print(f"  95th percentile: {metrics.p95_time:.3f}s")
            print(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/s")

    @pytest.mark.performance
    def test_knowledge_retrieval_performance(self, performance_rag_system, performance_profiler):
        """Test performance of knowledge retrieval operations"""
        km, agent_id = performance_rag_system

        # Pre-populate with test data
        num_entries = 1000
        for i in range(num_entries):
            content = f"Test knowledge entry {i} about topic {i % 20} with additional content"
            km.store_knowledge(
                agent_id=agent_id,
                content=content,
                metadata={"index": i, "topic": i % 20},
                tags=[f"topic_{i % 20}", "test_data"],
                source="perf_setup"
            )

        # Test different query patterns
        query_patterns = [
            ("specific_match", "Test knowledge entry 500"),
            ("topic_search", "topic 5 content"),
            ("general_search", "knowledge entry about"),
            ("fuzzy_search", "test additional information")
        ]

        for pattern_name, query in query_patterns:
            operation_name = f"search_{pattern_name}"

            # Run multiple searches
            for i in range(50):
                with performance_profiler.measure_operation(operation_name):
                    results = km.load_knowledge(
                        agent_id=agent_id,
                        query=query,
                        limit=10,
                        similarity_threshold=0.5
                    )
                    assert isinstance(results, list)

            # Analyze metrics
            metrics = performance_profiler.get_metrics(operation_name)

            # Performance assertions
            assert metrics.avg_time < 0.5, f"Average search time too high for {pattern_name}: {metrics.avg_time}s"
            assert metrics.p95_time < 1.0, f"95th percentile too high for {pattern_name}: {metrics.p95_time}s"

            print(f"\n{pattern_name.upper()} SEARCH PERFORMANCE:")
            print(f"  Average time: {metrics.avg_time:.3f}s")
            print(f"  95th percentile: {metrics.p95_time:.3f}s")
            print(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/s")

    @pytest.mark.performance
    def test_embedding_generation_performance(self, performance_profiler):
        """Test performance of embedding generation"""
        from test.rag_test_implementations import MockEmbeddingService

        embedding_service = MockEmbeddingService()

        # Test different text sizes
        text_sizes = [
            ("short", "Short text for embedding"),
            ("medium", "Medium length text " * 20),
            ("long", "Very long text content " * 100)
        ]

        for size_name, text in text_sizes:
            operation_name = f"embedding_{size_name}"

            # Generate multiple embeddings
            for i in range(100):
                test_text = f"{text} variation {i}"

                with performance_profiler.measure_operation(operation_name):
                    embedding = embedding_service.generate_embedding(test_text)
                    assert len(embedding) == 768

            # Analyze metrics
            metrics = performance_profiler.get_metrics(operation_name)

            # Performance assertions
            assert metrics.avg_time < 0.1, f"Embedding generation too slow for {size_name}: {metrics.avg_time}s"
            assert metrics.throughput_ops_per_sec > 50, f"Embedding throughput too low for {size_name}: {metrics.throughput_ops_per_sec} ops/s"

            print(f"\n{size_name.upper()} TEXT EMBEDDING PERFORMANCE:")
            print(f"  Average time: {metrics.avg_time:.4f}s")
            print(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/s")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_operations_performance(self, performance_rag_system, performance_profiler):
        """Test performance of batch operations"""
        km, agent_id = performance_rag_system

        # Test batch storage
        batch_sizes = [10, 50, 100, 200]

        for batch_size in batch_sizes:
            operation_name = f"batch_store_{batch_size}"

            # Prepare batch data
            batch_contents = [
                f"Batch entry {i} with content about topic {i % 10}"
                for i in range(batch_size)
            ]

            with performance_profiler.measure_operation(operation_name):
                stored_ids = []
                for content in batch_contents:
                    knowledge_id = km.store_knowledge(
                        agent_id=agent_id,
                        content=content,
                        metadata={"batch_size": batch_size},
                        tags=["batch_test"],
                        source="batch_perf"
                    )
                    stored_ids.append(knowledge_id)

                assert len(stored_ids) == batch_size

            # Analyze batch performance
            metrics = performance_profiler.get_metrics(operation_name)
            per_item_time = metrics.avg_time / batch_size

            print(f"\nBATCH SIZE {batch_size} PERFORMANCE:")
            print(f"  Total time: {metrics.avg_time:.3f}s")
            print(f"  Per item: {per_item_time:.4f}s")
            print(f"  Batch throughput: {batch_size / metrics.avg_time:.1f} items/s")

            # Performance should scale reasonably
            assert per_item_time < 0.1, f"Per-item time too high for batch {batch_size}: {per_item_time}s"


class TestRAGLoadTesting:
    """Load testing for RAG system under concurrent usage"""

    @pytest.mark.load
    def test_concurrent_knowledge_storage(self, performance_rag_system):
        """Test concurrent knowledge storage operations"""
        km, agent_id = performance_rag_system
        load_tester = RAGLoadTester(km, agent_id)

        def store_operation():
            content = f"Concurrent storage test {time.time()}"
            return km.store_knowledge(
                agent_id=agent_id,
                content=content,
                tags=["concurrent_test"],
                source="load_test"
            )

        # Configure load test
        config = LoadTestConfig(
            concurrent_users=10,
            operations_per_user=20,
            ramp_up_time=2.0,
            think_time=0.05
        )

        # Run load test
        start_time = time.time()
        results = load_tester.run_concurrent_operations(store_operation, config)
        end_time = time.time()

        # Analyze results
        analysis = load_tester.analyze_load_test_results(results)

        print(f"\nCONCURRENT STORAGE LOAD TEST RESULTS:")
        print(f"  Total operations: {analysis['summary']['total_operations']}")
        print(f"  Success rate: {analysis['summary']['success_rate_percent']:.1f}%")
        print(f"  Total duration: {end_time - start_time:.2f}s")
        print(f"  Throughput: {analysis['summary']['throughput_ops_per_sec']:.1f} ops/s")
        print(f"  Average response time: {analysis['timing_stats']['avg_duration']:.3f}s")
        print(f"  95th percentile: {analysis['timing_stats']['p95_duration']:.3f}s")

        # Performance assertions
        assert analysis['summary']['success_rate_percent'] >= 95.0
        assert analysis['timing_stats']['avg_duration'] < 1.0
        assert analysis['summary']['throughput_ops_per_sec'] > 10.0

        if analysis['errors']:
            print(f"  Errors: {analysis['errors']}")

    @pytest.mark.load
    def test_concurrent_knowledge_retrieval(self, performance_rag_system):
        """Test concurrent knowledge retrieval operations"""
        km, agent_id = performance_rag_system
        load_tester = RAGLoadTester(km, agent_id)

        # Pre-populate with data
        for i in range(500):
            km.store_knowledge(
                agent_id=agent_id,
                content=f"Searchable content {i} about topic {i % 25}",
                metadata={"index": i, "topic": i % 25},
                tags=[f"topic_{i % 25}"],
                source="load_test_data"
            )

        # Define search queries
        queries = [
            "searchable content about topic",
            "specific topic 5 information",
            "content with index number",
            "topic related knowledge",
            "general search query"
        ]

        def search_operation():
            query = queries[int(time.time() * 1000) % len(queries)]
            return km.load_knowledge(
                agent_id=agent_id,
                query=query,
                limit=5,
                similarity_threshold=0.5
            )

        # Configure load test
        config = LoadTestConfig(
            concurrent_users=15,
            operations_per_user=30,
            ramp_up_time=3.0,
            think_time=0.1
        )

        # Run load test
        start_time = time.time()
        results = load_tester.run_concurrent_operations(search_operation, config)
        end_time = time.time()

        # Analyze results
        analysis = load_tester.analyze_load_test_results(results)

        print(f"\nCONCURRENT SEARCH LOAD TEST RESULTS:")
        print(f"  Total operations: {analysis['summary']['total_operations']}")
        print(f"  Success rate: {analysis['summary']['success_rate_percent']:.1f}%")
        print(f"  Total duration: {end_time - start_time:.2f}s")
        print(f"  Throughput: {analysis['summary']['throughput_ops_per_sec']:.1f} ops/s")
        print(f"  Average response time: {analysis['timing_stats']['avg_duration']:.3f}s")
        print(f"  95th percentile: {analysis['timing_stats']['p95_duration']:.3f}s")

        # Performance assertions
        assert analysis['summary']['success_rate_percent'] >= 98.0
        assert analysis['timing_stats']['avg_duration'] < 0.5
        assert analysis['summary']['throughput_ops_per_sec'] > 20.0

    @pytest.mark.load
    def test_mixed_workload_performance(self, performance_rag_system):
        """Test performance under mixed read/write workload"""
        km, agent_id = performance_rag_system
        load_tester = RAGLoadTester(km, agent_id)

        # Pre-populate with initial data
        for i in range(100):
            km.store_knowledge(
                agent_id=agent_id,
                content=f"Initial content {i}",
                tags=["initial"],
                source="mixed_test_init"
            )

        def mixed_operation():
            """Randomly perform either storage or retrieval"""
            import random

            if random.random() < 0.3:  # 30% writes
                content = f"Mixed workload content {time.time()}"
                return km.store_knowledge(
                    agent_id=agent_id,
                    content=content,
                    tags=["mixed_test"],
                    source="mixed_workload"
                )
            else:  # 70% reads
                return km.load_knowledge(
                    agent_id=agent_id,
                    query="content information",
                    limit=5
                )

        # Configure mixed workload test
        config = LoadTestConfig(
            concurrent_users=12,
            operations_per_user=50,
            ramp_up_time=3.0,
            think_time=0.05
        )

        # Run load test
        start_time = time.time()
        results = load_tester.run_concurrent_operations(mixed_operation, config)
        end_time = time.time()

        # Analyze results
        analysis = load_tester.analyze_load_test_results(results)

        print(f"\nMIXED WORKLOAD LOAD TEST RESULTS:")
        print(f"  Total operations: {analysis['summary']['total_operations']}")
        print(f"  Success rate: {analysis['summary']['success_rate_percent']:.1f}%")
        print(f"  Total duration: {end_time - start_time:.2f}s")
        print(f"  Throughput: {analysis['summary']['throughput_ops_per_sec']:.1f} ops/s")
        print(f"  Average response time: {analysis['timing_stats']['avg_duration']:.3f}s")

        # Performance assertions for mixed workload
        assert analysis['summary']['success_rate_percent'] >= 95.0
        assert analysis['timing_stats']['avg_duration'] < 0.8
        assert analysis['summary']['throughput_ops_per_sec'] > 15.0


class TestRAGStressTesting:
    """Stress testing to find system limits"""

    @pytest.mark.stress
    def test_memory_usage_under_load(self, performance_rag_system):
        """Test memory usage under high load"""
        import gc

        km, agent_id = performance_rag_system

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Store increasing amounts of data
        batch_sizes = [100, 500, 1000, 2000]
        memory_usage = []

        for batch_size in batch_sizes:
            # Store batch of knowledge
            for i in range(batch_size):
                content = f"Memory test content {i} " * 50  # Larger content
                km.store_knowledge(
                    agent_id=agent_id,
                    content=content,
                    metadata={"batch": batch_size, "index": i},
                    tags=["memory_test"],
                    source="stress_test"
                )

            # Force garbage collection and measure memory
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            memory_usage.append(memory_increase)

            print(f"After {batch_size} entries: {memory_increase:.1f} MB increase")

            # Memory usage should not grow excessively
            assert memory_increase < batch_size * 0.1, f"Memory usage too high: {memory_increase:.1f} MB for {batch_size} entries"

        # Memory growth should be roughly linear
        if len(memory_usage) >= 2:
            growth_rates = [memory_usage[i] / memory_usage[i-1] for i in range(1, len(memory_usage))]
            avg_growth_rate = statistics.mean(growth_rates)
            assert avg_growth_rate < 3.0, f"Memory growth rate too high: {avg_growth_rate}"

    @pytest.mark.stress
    def test_large_document_processing(self, performance_rag_system):
        """Test processing of very large documents"""
        km, agent_id = performance_rag_system

        # Test with increasingly large documents
        document_sizes = [10000, 50000, 100000, 200000]  # Characters

        for size in document_sizes:
            # Create large document
            large_content = f"Large document content section. " * (size // 30)

            start_time = time.time()

            # Process large document (chunking would happen in real implementation)
            chunks = km.chunker.chunk_text(large_content, chunk_size=1000, chunk_overlap=200)

            # Store all chunks
            stored_ids = []
            for i, chunk in enumerate(chunks):
                knowledge_id = km.store_knowledge(
                    agent_id=agent_id,
                    content=chunk,
                    metadata={"document_size": size, "chunk_index": i},
                    tags=["large_doc", f"size_{size}"],
                    source="stress_test"
                )
                stored_ids.append(knowledge_id)

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"Document size {size:,} chars: {len(chunks)} chunks in {processing_time:.2f}s")

            # Performance assertions
            assert processing_time < 30.0, f"Processing time too high for {size} chars: {processing_time}s"
            assert len(stored_ids) == len(chunks), "Not all chunks were stored"

            # Test retrieval from large document
            search_start = time.time()
            results = km.load_knowledge(
                agent_id=agent_id,
                query="large document content section",
                limit=10,
                similarity_threshold=0.5
            )
            search_time = time.time() - search_start

            assert search_time < 2.0, f"Search time too high for large document: {search_time}s"
            assert len(results) > 0, "No results found for large document search"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short",
        "-s"  # Show print statements
    ])