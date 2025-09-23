#!/usr/bin/env python3
"""
RAG Testing Suite Runner
Comprehensive test execution script for RAG and Vector DB systems
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json

class TestRunner:
    """Main test runner for RAG system testing"""

    def __init__(self, project_root: str = None):
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Auto-detect project root from script location
            script_dir = Path(__file__).parent
            # If script is in test/ subdirectory, go up one level
            if script_dir.name == 'test':
                self.project_root = script_dir.parent
            elif script_dir.name == 'design':
                self.project_root = script_dir.parent
            else:
                self.project_root = script_dir

        self.test_results = {}
        self.start_time = None
        self.python_executable = self._detect_python_executable()

    def _detect_python_executable(self) -> str:
        """Detect the correct Python executable"""
        # First try virtual environment if available
        venv_python = self.project_root / '.venv' / 'bin' / 'python'
        if venv_python.exists():
            return str(venv_python)

        candidates = ['python3', 'python', 'python3.11', 'python3.10', 'python3.9']

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        # Fallback to python3 if nothing found
        return 'python3'

    def run_command(self, command: List[str], description: str = None) -> Dict[str, Any]:
        """Run a command and capture results"""
        if description:
            print(f"\n🔄 {description}")
            print(f"   Command: {' '.join(command)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            if success:
                print(f"   ✅ Completed in {duration:.2f}s")
            else:
                print(f"   ❌ Failed in {duration:.2f}s")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")

            return {
                'success': success,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout after 5 minutes")
            return {
                'success': False,
                'duration': time.time() - start_time,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out'
            }

        except Exception as e:
            print(f"   💥 Exception: {e}")
            return {
                'success': False,
                'duration': time.time() - start_time,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }

    def run_unit_tests(self) -> bool:
        """Run unit tests for all components"""
        print("\n" + "="*60)
        print("🧪 RUNNING UNIT TESTS")
        print("="*60)

        test_files = [
            "test/rag_test_implementations.py::TestEmbeddingService",
            "test/rag_test_implementations.py::TestVectorStore",
            "test/rag_test_implementations.py::TestKnowledgeManager",
            "test/rag_test_implementations.py::TestTextChunking"
        ]

        all_passed = True
        for test_file in test_files:
            component_name = test_file.split("::")[-1]
            result = self.run_command(
                [self.python_executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                f"Testing {component_name}"
            )
            self.test_results[f"unit_{component_name.lower()}"] = result
            if not result['success']:
                all_passed = False

        return all_passed

    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("\n" + "="*60)
        print("🔗 RUNNING INTEGRATION TESTS")
        print("="*60)

        result = self.run_command(
            [
                self.python_executable, "-m", "pytest",
                "test/rag_test_implementations.py::TestRAGIntegration",
                "-v", "--tb=short", "-m", "integration"
            ],
            "RAG Integration Tests"
        )

        self.test_results['integration_rag'] = result
        return result['success']

    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        print("\n" + "="*60)
        print("⚡ RUNNING PERFORMANCE TESTS")
        print("="*60)

        performance_tests = [
            ("Basic Performance", ["test/rag_performance_tests.py::TestRAGPerformance", "-m", "performance"]),
            ("Load Testing", ["test/rag_performance_tests.py::TestRAGLoadTesting", "-m", "load"]),
        ]

        all_passed = True
        for test_name, pytest_args in performance_tests:
            result = self.run_command(
                [self.python_executable, "-m", "pytest"] + pytest_args + ["-v", "--tb=short", "-s"],
                test_name
            )
            self.test_results[f"performance_{test_name.lower().replace(' ', '_')}"] = result
            if not result['success']:
                all_passed = False

        return all_passed

    def run_stress_tests(self) -> bool:
        """Run stress tests"""
        print("\n" + "="*60)
        print("💪 RUNNING STRESS TESTS")
        print("="*60)

        result = self.run_command(
            [
                self.python_executable, "-m", "pytest",
                "test/rag_performance_tests.py::TestRAGStressTesting",
                "-v", "--tb=short", "-m", "stress", "-s"
            ],
            "RAG Stress Tests"
        )

        self.test_results['stress_testing'] = result
        return result['success']

    def run_coverage_analysis(self) -> bool:
        """Run test coverage analysis"""
        print("\n" + "="*60)
        print("📊 RUNNING COVERAGE ANALYSIS")
        print("="*60)

        # Run tests with coverage
        result = self.run_command(
            [
                self.python_executable, "-m", "pytest",
                "test/rag_test_implementations.py",
                "--cov=rag",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-report=json",
                "-v"
            ],
            "Test Coverage Analysis"
        )

        self.test_results['coverage'] = result

        # Parse coverage results if available
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                    print(f"   📈 Total Coverage: {total_coverage:.1f}%")

                    if total_coverage >= 80:
                        print("   ✅ Coverage target met (≥80%)")
                    else:
                        print("   ⚠️  Coverage below target (<80%)")

            except Exception as e:
                print(f"   ⚠️  Could not parse coverage data: {e}")

        return result['success']

    def validate_environment(self) -> bool:
        """Validate test environment setup"""
        print("\n" + "="*60)
        print("🔧 VALIDATING TEST ENVIRONMENT")
        print("="*60)

        checks = [
            ([self.python_executable, "--version"], "Python Version"),
            ([self.python_executable, "-c", "import pytest; print(f'pytest {pytest.__version__}')"], "Pytest Installation"),
            ([self.python_executable, "-c", "import numpy; print(f'numpy {numpy.__version__}')"], "NumPy Installation"),
            ([self.python_executable, "-c", "import psutil; print(f'psutil {psutil.__version__}')"], "Psutil Installation"),
        ]

        all_valid = True
        for command, description in checks:
            result = self.run_command(command, description)
            if result['success'] and result['stdout']:
                print(f"   📋 {result['stdout'].strip()}")
            elif not result['success']:
                all_valid = False

        # Check for test files
        test_files = [
            "test/rag_test_implementations.py",
            "test/rag_performance_tests.py"
        ]

        for test_file in test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                print(f"   ✅ Found {test_file}")
            else:
                print(f"   ❌ Missing {test_file}")
                all_valid = False

        return all_valid

    def run_quick_validation(self) -> bool:
        """Run quick validation tests"""
        print("\n" + "="*60)
        print("🚀 RUNNING QUICK VALIDATION")
        print("="*60)

        # Run a subset of critical tests
        quick_tests = [
            "test/rag_test_implementations.py::TestEmbeddingService::test_mock_embedding_generation",
            "test/rag_test_implementations.py::TestVectorStore::test_collection_management",
            "test/rag_test_implementations.py::TestKnowledgeManager::test_knowledge_storage_and_retrieval",
            "test/rag_test_implementations.py::TestRAGIntegration::test_document_learning_workflow"
        ]

        result = self.run_command(
            [self.python_executable, "-m", "pytest"] + quick_tests + ["-v", "--tb=short"],
            "Quick Validation Tests"
        )

        self.test_results['quick_validation'] = result
        return result['success']

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time if self.start_time else 0

        report = []
        report.append("="*80)
        report.append("🧪 RAG SYSTEM TEST REPORT")
        report.append("="*80)
        report.append(f"📅 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        report.append("")

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('success', False))
        failed_tests = total_tests - passed_tests

        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Test Suites: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {(passed_tests / total_tests * 100) if total_tests > 0 else 0:.1f}%")
        report.append("")

        # Detailed Results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 40)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            duration = result.get('duration', 0)
            report.append(f"{status} {test_name:<30} ({duration:.2f}s)")

            if not result.get('success', False) and result.get('stderr'):
                # Include first few lines of error
                error_lines = result['stderr'].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        report.append(f"     Error: {line.strip()}")

        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        if failed_tests == 0:
            report.append("🎉 All tests passed! System is ready for deployment.")
        else:
            report.append("🔧 Issues found that need attention:")
            for test_name, result in self.test_results.items():
                if not result.get('success', False):
                    report.append(f"   • Fix {test_name} test failures")

        if total_duration > 300:  # 5 minutes
            report.append("⚡ Consider optimizing test performance:")
            report.append("   • Use more mocking for unit tests")
            report.append("   • Parallelize test execution")
            report.append("   • Review test data sizes")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def save_results(self, output_file: str = None):
        """Save test results to file"""
        if output_file is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = f"rag_test_report_{timestamp}.json"

        output_path = self.project_root / output_file

        # Prepare data for JSON serialization
        serializable_results = {}
        for test_name, result in self.test_results.items():
            serializable_results[test_name] = {
                'success': result.get('success', False),
                'duration': result.get('duration', 0),
                'returncode': result.get('returncode', -1),
                'has_output': len(result.get('stdout', '')) > 0,
                'has_errors': len(result.get('stderr', '')) > 0
            }

        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': time.time() - self.start_time if self.start_time else 0,
            'results': serializable_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results.values() if r.get('success', False)),
                'failed_tests': sum(1 for r in self.test_results.values() if not r.get('success', False))
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"📄 Test results saved to: {output_path}")


def main():
    """Main CLI interface for test runner"""
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "performance", "stress", "coverage", "quick"],
        default="quick",
        help="Test suite to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test results (JSON format)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating final report"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner()
    runner.start_time = time.time()

    print("🧪 RAG SYSTEM TEST RUNNER")
    print("=" * 50)

    # Validate environment first
    if not runner.validate_environment():
        print("\n❌ Environment validation failed. Please fix issues before running tests.")
        sys.exit(1)

    success = True

    try:
        if args.suite == "quick":
            success = runner.run_quick_validation()
        elif args.suite == "unit":
            success = runner.run_unit_tests()
        elif args.suite == "integration":
            success = runner.run_integration_tests()
        elif args.suite == "performance":
            success = runner.run_performance_tests()
        elif args.suite == "stress":
            success = runner.run_stress_tests()
        elif args.suite == "coverage":
            success = runner.run_coverage_analysis()
        elif args.suite == "all":
            # Run all test suites
            success = (
                runner.run_unit_tests() and
                runner.run_integration_tests() and
                runner.run_performance_tests() and
                runner.run_coverage_analysis()
            )

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        success = False

    # Generate and display report
    if not args.no_report:
        report = runner.generate_test_report()
        print("\n" + report)

    # Save results
    if args.output or len(runner.test_results) > 0:
        runner.save_results(args.output)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()