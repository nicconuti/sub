#!/usr/bin/env python3
"""
Systematic Debugging Framework Test Suite
Ensures that systematic debugging approaches are enforced across the entire application
"""

import asyncio
import json
import time
import logging
import unittest
from typing import Dict, Any, List
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystematicDebuggingTests(unittest.TestCase):
    """
    Test suite that validates systematic debugging patterns are followed
    This ensures we never go back to ad-hoc debugging approaches
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path("/Users/nicco.nuti/Documents/GitHub/sub")
        cls.backend_dir = cls.project_root / "backend"
        cls.frontend_dir = cls.project_root / "frontend/subwoofer-sim-ui"
        cls.tests_dir = cls.project_root / "tests"
        
    def test_error_responses_include_request_id(self):
        """
        CRITICAL: Test that ALL error responses include request_id for systematic debugging
        This prevents the timeout issues we experienced
        """
        logger.info("🧪 Testing Error Response request_id Inclusion")
        
        # Check backend websocket_manager.py
        ws_manager_file = self.backend_dir / "api/services/websocket_manager.py"
        self.assertTrue(ws_manager_file.exists(), "WebSocket manager file must exist")
        
        with open(ws_manager_file, 'r') as f:
            content = f.read()
            
        # Verify send_error method includes request_id parameter
        self.assertIn("def send_error(self, session_id: str, error_code: str, message: str, request_id: str = None):", 
                     content, "send_error must accept request_id parameter")
        
        # Verify request_id is included in error response
        self.assertIn('error_response["request_id"] = request_id', 
                     content, "Error responses must include request_id when provided")
        
        logger.info("✅ Error response request_id inclusion validated")
    
    def test_main_py_extracts_request_id_from_invalid_events(self):
        """
        CRITICAL: Test that main.py extracts request_id even from invalid events
        This ensures proper error matching
        """
        logger.info("🧪 Testing main.py Invalid Event request_id Extraction")
        
        main_file = self.backend_dir / "main.py"
        self.assertTrue(main_file.exists(), "main.py must exist")
        
        with open(main_file, 'r') as f:
            content = f.read()
            
        # Verify that invalid event handling tries to extract request_id
        self.assertIn("request_id = event_data.get('request_id')", 
                     content, "Must extract request_id from invalid events")
        
        # Verify request_id is passed to send_error
        self.assertIn('await websocket_manager.send_error(session_id, "invalid_event", str(e), request_id)',
                     content, "Invalid event errors must include extracted request_id")
        
        logger.info("✅ main.py request_id extraction validated")
    
    def test_frontend_websocket_timeout_handling(self):
        """
        CRITICAL: Test that frontend has proper timeout handling with request_id matching
        """
        logger.info("🧪 Testing Frontend WebSocket Timeout Handling")
        
        connection_file = self.frontend_dir / "src/store/connection.ts"
        self.assertTrue(connection_file.exists(), "Connection store must exist")
        
        with open(connection_file, 'r') as f:
            content = f.read()
            
        # Verify timeout handling exists
        self.assertIn("timeout", content.lower(), "Must have timeout handling")
        
        # Verify request_id matching
        self.assertIn("request_id", content, "Must use request_id for matching")
        
        # Verify Promise-based error handling
        self.assertIn("Promise", content, "Must use Promise for async handling")
        self.assertIn("reject", content, "Must use reject for timeout errors")
        
        logger.info("✅ Frontend timeout handling validated")
    
    def test_spl_data_safe_processing(self):
        """
        CRITICAL: Test that SPL data processing uses safe methods to prevent stack overflow
        """
        logger.info("🧪 Testing SPL Data Safe Processing")
        
        # Check for safe flattening patterns
        files_to_check = [
            self.frontend_dir / "src/store/simulation.ts",
            self.frontend_dir / "src/components/SplViewer.tsx"
        ]
        
        unsafe_patterns = [
            "Math.min(...",  # Spread operator with large arrays
            "Math.max(...",  # Spread operator with large arrays
            ".flat()",       # Recursive flattening
        ]
        
        safe_patterns = [
            "forEach",       # Safe iteration
            "isFinite",      # Safe numeric validation
            "Array.isArray", # Type checking
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for unsafe patterns
                for pattern in unsafe_patterns:
                    self.assertNotIn(pattern, content, 
                                   f"File {file_path.name} should not use unsafe pattern: {pattern}")
                
                # Check for at least some safe patterns
                safe_count = sum(1 for pattern in safe_patterns if pattern in content)
                self.assertGreater(safe_count, 0, 
                                 f"File {file_path.name} should use safe processing patterns")
        
        logger.info("✅ SPL data safe processing validated")
    
    def test_comprehensive_test_coverage(self):
        """Test that comprehensive test files exist and are maintained"""
        logger.info("🧪 Testing Comprehensive Test Coverage")
        
        required_test_files = [
            "test_websocket_timeout.py",
            "test_comprehensive_app.py", 
            "test_frontend_systematic.py",
            "test_systematic_debugging.py"
        ]
        
        for test_file in required_test_files:
            test_path = self.tests_dir / test_file
            self.assertTrue(test_path.exists(), f"Test file {test_file} must exist")
            
            # Check file is not empty
            with open(test_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 1000, 
                                 f"Test file {test_file} should have substantial content")
        
        logger.info("✅ Comprehensive test coverage validated")
    
    def test_error_logging_patterns(self):
        """Test that proper error logging patterns are used throughout"""
        logger.info("🧪 Testing Error Logging Patterns")
        
        # Check backend files for proper logging
        backend_files = [
            self.backend_dir / "main.py",
            self.backend_dir / "api/services/websocket_manager.py",
            self.backend_dir / "api/services/simulation_service.py"
        ]
        
        required_log_patterns = [
            "logger.error",
            "logger.warning", 
            "logger.info"
        ]
        
        for file_path in backend_files:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                log_pattern_count = sum(1 for pattern in required_log_patterns if pattern in content)
                self.assertGreater(log_pattern_count, 0, 
                                 f"File {file_path.name} should have proper logging")
        
        logger.info("✅ Error logging patterns validated")
    
    def test_type_safety_patterns(self):
        """Test that type safety patterns are maintained"""
        logger.info("🧪 Testing Type Safety Patterns")
        
        # Check TypeScript files for type safety
        ts_files = []
        if self.frontend_dir.exists():
            ts_files = list(self.frontend_dir.glob("src/**/*.ts")) + list(self.frontend_dir.glob("src/**/*.tsx"))
        
        type_safety_patterns = [
            ": string",
            ": number", 
            ": boolean",
            "interface ",
            "type ",
            "Dict[",
            "List[",
            "Optional[",
        ]
        
        # Check at least some files have type annotations
        typed_files = 0
        for file_path in ts_files[:5]:  # Check first 5 files
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                type_count = sum(1 for pattern in type_safety_patterns if pattern in content)
                if type_count > 0:
                    typed_files += 1
        
        if ts_files:
            self.assertGreater(typed_files, 0, "Some files should have type annotations")
        
        logger.info("✅ Type safety patterns validated")
    
    def test_performance_safeguards(self):
        """Test that performance safeguards are in place"""
        logger.info("🧪 Testing Performance Safeguards")
        
        # Check for performance-related patterns
        files_to_check = [
            self.frontend_dir / "src/store/simulation.ts",
            self.backend_dir / "api/services/websocket_manager.py"
        ]
        
        performance_patterns = [
            "timeout",
            "limit",
            "chunk",
            "sample",
            "max",
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    
                perf_count = sum(1 for pattern in performance_patterns if pattern in content)
                self.assertGreater(perf_count, 0, 
                                 f"File {file_path.name} should have performance safeguards")
        
        logger.info("✅ Performance safeguards validated")

def run_systematic_debugging_validation():
    """Run the systematic debugging validation test suite"""
    logger.info("🚀 Starting Systematic Debugging Validation")
    logger.info("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SystematicDebuggingTests)
    
    # Run tests with custom result handler
    class SystematicTestResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.systematic_failures = []
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.systematic_failures.append((test, err))
            logger.error(f"❌ SYSTEMATIC FAILURE: {test._testMethodName}")
            
        def addError(self, test, err):
            super().addError(test, err)
            self.systematic_failures.append((test, err))
            logger.error(f"❌ SYSTEMATIC ERROR: {test._testMethodName}")
            
        def addSuccess(self, test):
            super().addSuccess(test)
            logger.info(f"✅ SYSTEMATIC PASS: {test._testMethodName}")
    
    # Run tests
    runner = unittest.TextTestRunner(
        resultclass=SystematicTestResult,
        verbosity=2,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    
    # Generate report
    logger.info("\n" + "=" * 70)
    logger.info("📊 SYSTEMATIC DEBUGGING VALIDATION RESULTS")
    logger.info("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures) + len(result.errors)
    passed = total_tests - failures
    
    logger.info(f"🎯 Tests Run: {total_tests}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failures}")
    
    if failures == 0:
        logger.info("🎉 PERFECT: All systematic debugging patterns are in place!")
        logger.info("🛡️ The application is protected against ad-hoc debugging approaches")
    else:
        logger.error("❌ SYSTEMATIC FAILURES DETECTED!")
        logger.error("🚨 The application needs systematic debugging improvements")
        
        # Log specific failures
        if hasattr(result, 'systematic_failures'):
            for test, error in result.systematic_failures:
                logger.error(f"  ❌ {test._testMethodName}: {error[1]}")
    
    return failures == 0

if __name__ == "__main__":
    success = run_systematic_debugging_validation()
    exit(0 if success else 1)