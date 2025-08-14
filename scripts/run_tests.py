#!/usr/bin/env python3
"""
Test execution script for NASA CMR Agent.

Provides comprehensive test running capabilities with different test categories,
coverage reporting, and performance benchmarking.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for NASA CMR Agent with comprehensive options."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.activate_cmd = f"source {self.project_root}/.tech/bin/activate"
        
    def run_command(self, cmd: str, description: str = None) -> int:
        """Run a shell command with proper error handling."""
        if description:
            print(f"\n🔄 {description}")
            print("=" * 50)
        
        full_cmd = f"{self.activate_cmd} && cd {self.project_root} && {cmd}"
        result = subprocess.run(full_cmd, shell=True, executable='/bin/bash')
        
        return result.returncode
    
    def run_unit_tests(self, verbose: bool = True, coverage: bool = True) -> int:
        """Run unit tests."""
        cmd = ["python", "-m", "pytest", "tests/unit/"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=nasa_cmr_agent",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml"
            ])
        else:
            cmd.append("--no-cov")
        
        # Add specific markers for unit tests
        cmd.extend(["-m", "unit"])
        
        return self.run_command(" ".join(cmd), "Running Unit Tests")
    
    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests."""
        cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(" ".join(cmd), "Running Integration Tests")
    
    def run_performance_tests(self, verbose: bool = True, include_slow: bool = False) -> int:
        """Run performance tests."""
        cmd = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
        
        if verbose:
            cmd.append("-v")
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        return self.run_command(" ".join(cmd), "Running Performance Tests")
    
    def run_specific_test(self, test_path: str, verbose: bool = True) -> int:
        """Run specific test file or test function."""
        cmd = ["python", "-m", "pytest", test_path]
        
        if verbose:
            cmd.append("-v")
        
        return self.run_command(" ".join(cmd), f"Running Specific Test: {test_path}")
    
    def run_all_tests(self, include_performance: bool = False, include_slow: bool = False) -> int:
        """Run all tests in sequence."""
        results = []
        
        # Run unit tests
        result = self.run_unit_tests()
        results.append(("Unit Tests", result))
        
        # Run integration tests  
        result = self.run_integration_tests()
        results.append(("Integration Tests", result))
        
        # Run performance tests if requested
        if include_performance:
            result = self.run_performance_tests(include_slow=include_slow)
            results.append(("Performance Tests", result))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        overall_success = True
        for test_type, return_code in results:
            status = "✅ PASSED" if return_code == 0 else "❌ FAILED"
            print(f"{test_type:.<30} {status}")
            if return_code != 0:
                overall_success = False
        
        if overall_success:
            print("\n🎉 All tests passed successfully!")
            return 0
        else:
            print("\n💥 Some tests failed. Check output above.")
            return 1
    
    def run_quick_tests(self) -> int:
        """Run quick smoke tests for rapid feedback."""
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/test_models_schemas.py::TestSpatialConstraint::test_valid_spatial_constraint",
            "tests/unit/test_circuit_breaker.py::TestCircuitBreakerService::test_circuit_breaker_initialization", 
            "-v", "--tb=short"
        ]
        
        return self.run_command(" ".join(cmd), "Running Quick Smoke Tests")
    
    def run_coverage_report(self) -> int:
        """Generate and display coverage report."""
        cmd = [
            "python", "-m", "pytest", 
            "--cov=nasa_cmr_agent",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-only"
        ]
        
        return self.run_command(" ".join(cmd), "Generating Coverage Report")
    
    def lint_code(self) -> int:
        """Run code linting."""
        results = []
        
        # Run black
        result = self.run_command("python -m black --check .", "Code Formatting Check (Black)")
        results.append(("Black", result))
        
        # Run isort
        result = self.run_command("python -m isort --check-only .", "Import Sorting Check (isort)")
        results.append(("isort", result))
        
        # Run flake8
        result = self.run_command("python -m flake8 .", "Code Style Check (flake8)")
        results.append(("flake8", result))
        
        # Run mypy
        result = self.run_command("python -m mypy nasa_cmr_agent", "Type Checking (mypy)")
        results.append(("mypy", result))
        
        # Summary
        print("\n" + "=" * 60)
        print("LINTING SUMMARY")
        print("=" * 60)
        
        overall_success = True
        for tool, return_code in results:
            status = "✅ PASSED" if return_code == 0 else "❌ FAILED"
            print(f"{tool:.<20} {status}")
            if return_code != 0:
                overall_success = False
        
        return 0 if overall_success else 1


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for NASA CMR Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "performance", "all", 
            "quick", "coverage", "lint", "specific"
        ],
        help="Test category to run"
    )
    
    parser.add_argument(
        "--test-path",
        help="Specific test path for 'specific' command"
    )
    
    parser.add_argument(
        "--no-verbose", "-q",
        action="store_true",
        help="Disable verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true", 
        help="Disable coverage reporting for unit tests"
    )
    
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Include performance tests in 'all' command"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests in performance testing"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    verbose = not args.no_verbose
    
    try:
        if args.command == "unit":
            return runner.run_unit_tests(verbose=verbose, coverage=not args.no_coverage)
        
        elif args.command == "integration":
            return runner.run_integration_tests(verbose=verbose)
        
        elif args.command == "performance":
            return runner.run_performance_tests(verbose=verbose, include_slow=args.include_slow)
        
        elif args.command == "all":
            return runner.run_all_tests(
                include_performance=args.include_performance,
                include_slow=args.include_slow
            )
        
        elif args.command == "quick":
            return runner.run_quick_tests()
        
        elif args.command == "coverage":
            return runner.run_coverage_report()
        
        elif args.command == "lint":
            return runner.lint_code()
        
        elif args.command == "specific":
            if not args.test_path:
                print("❌ Error: --test-path is required for 'specific' command")
                return 1
            return runner.run_specific_test(args.test_path, verbose=verbose)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())