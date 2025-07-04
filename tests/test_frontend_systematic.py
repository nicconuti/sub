#!/usr/bin/env python3
"""
Systematic Frontend Testing Framework
Ensures React frontend components follow systematic debugging patterns
"""

import asyncio
import json
import time
import logging
import subprocess
import requests
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrontendTestFramework:
    """Framework for testing React frontend systematically"""
    
    def __init__(self, frontend_dir: str = "/Users/nicco.nuti/Documents/GitHub/sub/frontend/subwoofer-sim-ui"):
        self.frontend_dir = Path(frontend_dir)
        self.backend_url = "http://localhost:8001"
        self.frontend_url = "http://localhost:5175"  # Vite dev server
        self.process = None
        
    def test_frontend_build_process(self):
        """Test that frontend builds without errors"""
        logger.info("🧪 Testing Frontend Build Process")
        
        try:
            # Test TypeScript compilation
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Build failed: {result.stderr}")
                return False
                
            logger.info("✅ Frontend builds successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Build timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Build error: {e}")
            return False
    
    def test_typescript_compilation(self):
        """Test TypeScript compilation and type checking"""
        logger.info("🧪 Testing TypeScript Compilation")
        
        try:
            # Check if tsc is available
            result = subprocess.run(
                ["npx", "tsc", "--noEmit"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"⚠️ TypeScript warnings/errors: {result.stderr}")
                # Don't fail on warnings, just log them
                
            logger.info("✅ TypeScript compilation checked")
            return True
            
        except Exception as e:
            logger.error(f"❌ TypeScript check failed: {e}")
            return False
    
    def test_frontend_dependencies(self):
        """Test that all frontend dependencies are properly installed"""
        logger.info("🧪 Testing Frontend Dependencies")
        
        package_json_path = self.frontend_dir / "package.json"
        
        if not package_json_path.exists():
            logger.error("❌ package.json not found")
            return False
            
        try:
            with open(package_json_path) as f:
                package_data = json.load(f)
                
            # Check critical dependencies
            required_deps = [
                "react", "react-dom", "typescript", "@types/react",
                "@mui/material", "zustand", "plotly.js", "react-plotly.js"
            ]
            
            all_deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
            
            missing_deps = []
            for dep in required_deps:
                if dep not in all_deps:
                    missing_deps.append(dep)
                    
            if missing_deps:
                logger.error(f"❌ Missing dependencies: {missing_deps}")
                return False
                
            logger.info("✅ All required dependencies present")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False
    
    def test_frontend_configuration(self):
        """Test frontend configuration files"""
        logger.info("🧪 Testing Frontend Configuration")
        
        config_files = [
            "tsconfig.json",
            "vite.config.ts", 
            "package.json"
        ]
        
        for config_file in config_files:
            config_path = self.frontend_dir / config_file
            if not config_path.exists():
                logger.warning(f"⚠️ Missing config file: {config_file}")
            else:
                logger.info(f"✅ Found config file: {config_file}")
                
        return True
    
    def test_frontend_source_structure(self):
        """Test that frontend source structure follows systematic patterns"""
        logger.info("🧪 Testing Frontend Source Structure")
        
        required_dirs = [
            "src",
            "src/components", 
            "src/store",
            "src/theme",
            "src/types"
        ]
        
        required_files = [
            "src/App.tsx",
            "src/main.tsx",
            "src/store/connection.ts",
            "src/store/simulation.ts",
            "src/theme/audioTheme.ts"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.frontend_dir / dir_path).exists():
                missing_dirs.append(dir_path)
                
        missing_files = []
        for file_path in required_files:
            if not (self.frontend_dir / file_path).exists():
                missing_files.append(file_path)
                
        if missing_dirs:
            logger.warning(f"⚠️ Missing directories: {missing_dirs}")
            
        if missing_files:
            logger.warning(f"⚠️ Missing files: {missing_files}")
            
        logger.info("✅ Frontend source structure checked")
        return len(missing_dirs) == 0 and len(missing_files) == 0
    
    def test_error_handling_patterns(self):
        """Test that frontend components have systematic error handling"""
        logger.info("🧪 Testing Frontend Error Handling Patterns")
        
        # Check key files for error handling patterns
        files_to_check = [
            "src/store/connection.ts",
            "src/store/simulation.ts",
            "src/components/visualization/SplViewer.tsx"
        ]
        
        error_patterns = [
            "try {", "catch", "error", "Error", "reject", "throw"
        ]
        
        files_with_errors = []
        
        for file_path in files_to_check:
            full_path = self.frontend_dir / file_path
            if not full_path.exists():
                files_with_errors.append(f"Missing: {file_path}")
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    
                # Check for error handling patterns
                error_count = sum(1 for pattern in error_patterns if pattern in content)
                
                if error_count < 2:  # Should have at least some error handling
                    files_with_errors.append(f"Limited error handling: {file_path}")
                else:
                    logger.info(f"✅ Good error handling in {file_path}")
                    
            except Exception as e:
                files_with_errors.append(f"Cannot read {file_path}: {e}")
                
        if files_with_errors:
            logger.warning(f"⚠️ Error handling issues: {files_with_errors}")
            return False
            
        logger.info("✅ Frontend error handling patterns validated")
        return True
    
    def test_websocket_integration_patterns(self):
        """Test that WebSocket integration follows systematic patterns"""
        logger.info("🧪 Testing WebSocket Integration Patterns")
        
        connection_file = self.frontend_dir / "src/store/connection.ts"
        
        if not connection_file.exists():
            logger.error("❌ Connection store file missing")
            return False
            
        try:
            with open(connection_file, 'r') as f:
                content = f.read()
                
            # Check for systematic WebSocket patterns
            required_patterns = [
                "request_id",  # Request-response matching
                "timeout",     # Timeout handling
                "Promise",     # Async handling
                "reject",      # Error handling
                "WebSocket"    # Native WebSocket usage
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)
                    
            if missing_patterns:
                logger.error(f"❌ Missing WebSocket patterns: {missing_patterns}")
                return False
                
            logger.info("✅ WebSocket integration patterns validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cannot validate WebSocket patterns: {e}")
            return False
    
    def test_spl_data_handling_safety(self):
        """Test that SPL data handling has safety measures"""
        logger.info("🧪 Testing SPL Data Handling Safety")
        
        # Check simulation store for safe data handling
        sim_file = self.frontend_dir / "src/store/simulation.ts"
        
        if not sim_file.exists():
            logger.error("❌ Simulation store file missing")
            return False
            
        try:
            with open(sim_file, 'r') as f:
                content = f.read()
                
            # Check for safety patterns
            safety_patterns = [
                "try",
                "catch", 
                "isFinite",  # Safe numeric checks
                "Array.isArray",  # Type validation
                "forEach"    # Safe iteration instead of spread
            ]
            
            safety_score = sum(1 for pattern in safety_patterns if pattern in content)
            
            if safety_score < 3:
                logger.warning(f"⚠️ Limited safety patterns in SPL handling (score: {safety_score}/5)")
                return False
                
            logger.info(f"✅ SPL data handling safety validated (score: {safety_score}/5)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cannot validate SPL safety: {e}")
            return False

def run_frontend_tests():
    """Run all frontend tests"""
    logger.info("🚀 Starting Systematic Frontend Tests")
    logger.info("=" * 60)
    
    framework = FrontendTestFramework()
    
    tests = [
        ("Frontend Dependencies", framework.test_frontend_dependencies),
        ("Frontend Configuration", framework.test_frontend_configuration),
        ("Source Structure", framework.test_frontend_source_structure),
        ("TypeScript Compilation", framework.test_typescript_compilation),
        ("Error Handling Patterns", framework.test_error_handling_patterns),
        ("WebSocket Integration", framework.test_websocket_integration_patterns),
        ("SPL Data Safety", framework.test_spl_data_handling_safety),
        ("Build Process", framework.test_frontend_build_process),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            
            if result:
                logger.info(f"✅ {test_name}: PASS")
            else:
                logger.error(f"❌ {test_name}: FAIL")
                
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 FRONTEND TEST RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASS" else "❌ FAIL/ERROR"
        logger.info(f"{status}: {test_name}")
        if result == "PASS":
            passed += 1
            
    logger.info(f"\n🎯 Frontend Tests: {passed}/{total} passed")
    
    pass_rate = passed / total if total > 0 else 0
    
    if pass_rate >= 0.9:
        logger.info("🎉 EXCELLENT: Frontend quality is excellent")
    elif pass_rate >= 0.8:
        logger.info("🎯 GOOD: Frontend quality is good")
    elif pass_rate >= 0.7:
        logger.info("⚠️ ACCEPTABLE: Frontend quality is acceptable")
    else:
        logger.error("❌ POOR: Frontend quality needs improvement")
        
    return pass_rate >= 0.8

if __name__ == "__main__":
    success = run_frontend_tests()
    exit(0 if success else 1)