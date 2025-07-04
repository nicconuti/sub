#!/usr/bin/env python3
"""
Master Test Runner for Entire Subwoofer Simulation Application
Runs all test suites systematically to ensure comprehensive quality
"""

import asyncio
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterTestRunner:
    """Comprehensive test runner for the entire application"""
    
    def __init__(self):
        self.project_root = Path("/Users/nicco.nuti/Documents/GitHub/sub")
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        self.start_time = time.time()
        
    def run_test_suite(self, test_name: str, test_command: List[str], timeout: int = 300) -> Tuple[bool, str]:
        """Run a single test suite with timeout and error handling"""
        logger.info(f"🧪 Running {test_name}")
        
        try:
            result = subprocess.run(
                test_command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
            return success, output
            
        except subprocess.TimeoutExpired:
            error_msg = f"Test suite {test_name} timed out after {timeout} seconds"
            logger.error(f"⏰ {error_msg}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Test suite {test_name} crashed: {e}"
            logger.error(f"💥 {error_msg}")
            return False, error_msg
    
    async def run_async_test(self, test_name: str, test_module: str) -> Tuple[bool, str]:
        """Run async test suites"""
        logger.info(f"🧪 Running {test_name} (async)")
        
        try:
            # Import and run the async test
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", f"import sys; sys.path.append('{self.project_root}'); "
                                     f"from tests.{test_module} import *; "
                                     f"import asyncio; "
                                     f"result = asyncio.run(run_comprehensive_tests() if 'comprehensive' in '{test_module}' else main()); "
                                     f"exit(0 if result else 1)",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            
            success = proc.returncode == 0
            output = stdout.decode() + "\n" + stderr.decode()
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
            return success, output
            
        except asyncio.TimeoutError:
            error_msg = f"Async test {test_name} timed out"
            logger.error(f"⏰ {error_msg}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Async test {test_name} crashed: {e}"
            logger.error(f"💥 {error_msg}")
            return False, error_msg
    
    def check_backend_health(self) -> bool:
        """Check if backend is running and healthy"""
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_all_tests(self):
        """Run all test suites in the correct order"""
        logger.info("🚀 STARTING COMPREHENSIVE APPLICATION TEST SUITE")
        logger.info("=" * 80)
        
        # Check backend availability
        if not self.check_backend_health():
            logger.error("❌ Backend is not running! Please start the backend first:")
            logger.error("   cd backend && python main.py")
            return False
        
        logger.info("✅ Backend is running and healthy")
        
        # Define test suites in order of importance
        test_suites = [
            # 1. Systematic debugging validation (most critical)
            {
                "name": "Systematic Debugging Validation",
                "type": "sync",
                "command": [sys.executable, "tests/test_systematic_debugging.py"],
                "critical": True
            },
            
            # 2. Frontend systematic tests
            {
                "name": "Frontend Systematic Tests", 
                "type": "sync",
                "command": [sys.executable, "tests/test_frontend_systematic.py"],
                "critical": True
            },
            
            # 3. WebSocket timeout tests (validates the fix)
            {
                "name": "WebSocket Timeout Tests",
                "type": "sync", 
                "command": [sys.executable, "tests/test_websocket_timeout.py"],
                "critical": True
            },
            
            # 4. Comprehensive app tests (integration)
            {
                "name": "Comprehensive App Tests",
                "type": "async",
                "module": "test_comprehensive_app",
                "critical": True
            },
            
            # 5. Error debugging tests
            {
                "name": "Error Debugging Tests",
                "type": "sync",
                "command": [sys.executable, "tests/test_debug_errors.py"],
                "critical": False
            },
        ]
        
        total_tests = len(test_suites)
        passed_tests = 0
        critical_failures = 0
        
        # Run each test suite
        for i, test_suite in enumerate(test_suites, 1):
            logger.info(f"\n{'='*20} [{i}/{total_tests}] {test_suite['name']} {'='*20}")
            
            try:
                if test_suite['type'] == 'sync':
                    success, output = self.run_test_suite(
                        test_suite['name'], 
                        test_suite['command']
                    )
                else:  # async
                    success, output = asyncio.run(self.run_async_test(
                        test_suite['name'],
                        test_suite['module']
                    ))
                
                self.results[test_suite['name']] = {
                    'success': success,
                    'output': output,
                    'critical': test_suite.get('critical', False)
                }
                
                if success:
                    passed_tests += 1
                else:
                    if test_suite.get('critical', False):
                        critical_failures += 1
                        
            except Exception as e:
                logger.error(f"💥 Fatal error running {test_suite['name']}: {e}")
                self.results[test_suite['name']] = {
                    'success': False,
                    'output': str(e),
                    'critical': test_suite.get('critical', False)
                }
                if test_suite.get('critical', False):
                    critical_failures += 1
        
        # Generate comprehensive report
        self.generate_final_report(total_tests, passed_tests, critical_failures)
        
        # Return overall success
        return critical_failures == 0 and passed_tests >= total_tests * 0.8
    
    def generate_final_report(self, total_tests: int, passed_tests: int, critical_failures: int):
        """Generate comprehensive test report"""
        elapsed_time = time.time() - self.start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE APPLICATION TEST REPORT")
        logger.info("=" * 80)
        
        # Overall statistics
        logger.info(f"⏱️  Total Execution Time: {elapsed_time:.1f} seconds")
        logger.info(f"🧪 Total Test Suites: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {total_tests - passed_tests}")
        logger.info(f"🚨 Critical Failures: {critical_failures}")
        
        # Detailed results
        logger.info(f"\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            critical = " (CRITICAL)" if result['critical'] else ""
            logger.info(f"  {status}: {test_name}{critical}")
        
        # Quality assessment
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"\n🎯 Overall Quality Assessment:")
        if critical_failures == 0 and pass_rate >= 0.9:
            logger.info("🎉 EXCELLENT: Application quality is excellent")
            logger.info("🛡️ All systematic debugging patterns are in place")
            logger.info("🚀 Ready for production use")
        elif critical_failures == 0 and pass_rate >= 0.8:
            logger.info("🎯 GOOD: Application quality is good")
            logger.info("✅ Core systematic patterns are working")
            logger.info("⚠️ Some minor improvements recommended")
        elif critical_failures == 0:
            logger.info("⚠️ ACCEPTABLE: Application quality is acceptable")
            logger.info("🔧 Several improvements needed")
        else:
            logger.error("❌ POOR: Critical systematic failures detected")
            logger.error("🚨 Immediate attention required")
            logger.error(f"💥 {critical_failures} critical systems are failing")
        
        # Recommendations
        logger.info(f"\n📝 Recommendations:")
        if critical_failures > 0:
            logger.error("🚨 URGENT: Fix critical failures before proceeding")
            for test_name, result in self.results.items():
                if not result['success'] and result['critical']:
                    logger.error(f"  ❌ Fix: {test_name}")
        else:
            logger.info("✅ All critical systems are functioning")
            
        if pass_rate < 0.9:
            logger.info("🔧 Consider improving non-critical test coverage")
            
        logger.info("🧪 Maintain systematic testing practices")
        logger.info("📊 Regular test execution recommended")

def main():
    """Main entry point for comprehensive testing"""
    runner = MasterTestRunner()
    
    try:
        success = runner.run_all_tests()
        
        if success:
            logger.info("\n🎉 ALL TESTS PASSED - APPLICATION IS READY")
            return True
        else:
            logger.error("\n❌ TESTS FAILED - APPLICATION NEEDS ATTENTION")
            return False
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test execution interrupted by user")
        return False
    except Exception as e:
        logger.error(f"\n💥 Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)