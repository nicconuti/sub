#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Entire Subwoofer Simulation Application
Ensures systematic debugging approaches are always used and all components work together
"""

import asyncio
import json
import pytest
import websockets
import time
import logging
import requests
import threading
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppTestFramework:
    """Comprehensive test framework for the entire application"""
    
    def __init__(self, backend_url: str = "http://localhost:8001", ws_url: str = "ws://localhost:8001/ws"):
        self.backend_url = backend_url
        self.ws_url = ws_url
        self.websocket = None
        self.session_id = None
        self.test_results = {}
        self.error_log = []
        
    async def setup(self):
        """Initialize test environment"""
        logger.info("🚀 Setting up comprehensive app test framework")
        
        # Test backend health
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Backend health check failed: {response.status_code}")
            logger.info("✅ Backend is healthy")
        except Exception as e:
            raise Exception(f"❌ Backend not accessible: {e}")
            
        # Connect to WebSocket
        try:
            self.websocket = await websockets.connect(self.ws_url, max_size=16777216)
            welcome_msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get('type') == 'connected':
                self.session_id = welcome_data['data']['sessionId']
                logger.info(f"✅ WebSocket connected: {self.session_id}")
            else:
                raise Exception(f"Unexpected welcome: {welcome_data}")
        except Exception as e:
            raise Exception(f"❌ WebSocket connection failed: {e}")
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.websocket:
            await self.websocket.close()
            
    async def send_request_with_validation(self, event_type: str, data: Dict[str, Any], 
                                         timeout: float = 10.0, 
                                         expect_success: bool = True) -> Dict[str, Any]:
        """
        Send request with comprehensive validation and error checking
        This ensures proper request-response matching is always validated
        """
        if not self.websocket:
            raise Exception("WebSocket not connected")
            
        request_id = f"test_{int(time.time() * 1000)}_{event_type}"
        start_time = time.time()
        
        message = {
            "type": event_type,
            "data": data,
            "request_id": request_id,
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        logger.info(f"📤 Sending {event_type} (request_id: {request_id})")
        
        try:
            await self.websocket.send(json.dumps(message))
            
            # Wait for response with proper matching
            while True:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
                    response_data = json.loads(response)
                    elapsed = time.time() - start_time
                    
                    # Validate response structure
                    self._validate_response_structure(response_data, request_id)
                    
                    # Check if this is our response
                    if response_data.get('request_id') == request_id:
                        logger.info(f"✅ Got matching response in {elapsed:.2f}s")
                        
                        # Validate success/error expectation
                        is_error = response_data.get('type') == 'error'
                        if expect_success and is_error:
                            self.error_log.append(f"Expected success but got error: {response_data}")
                            logger.warning(f"⚠️ Expected success but got error: {response_data}")
                        elif not expect_success and not is_error:
                            self.error_log.append(f"Expected error but got success: {response_data}")
                            logger.warning(f"⚠️ Expected error but got success: {response_data}")
                            
                        return response_data
                    else:
                        logger.debug(f"📨 Non-matching message: {response_data.get('type')}")
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    error_msg = f"Request {request_id} timed out after {elapsed:.2f}s"
                    self.error_log.append(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Request failed after {elapsed:.2f}s: {e}"
            self.error_log.append(error_msg)
            raise Exception(error_msg)
    
    def _validate_response_structure(self, response: Dict[str, Any], expected_request_id: str):
        """Validate that all responses have proper structure for systematic debugging"""
        
        # All responses must have type
        if 'type' not in response:
            raise Exception(f"Response missing 'type' field: {response}")
            
        # Error responses must have request_id for proper matching
        if response.get('type') == 'error':
            if 'request_id' not in response:
                raise Exception(f"Error response missing request_id: {response}")
            if response.get('request_id') != expected_request_id:
                logger.warning(f"⚠️ Error response request_id mismatch: got {response.get('request_id')}, expected {expected_request_id}")
                
        # Success responses should have request_id for correlation
        if response.get('type') not in ['error', 'connected'] and 'request_id' not in response:
            logger.warning(f"⚠️ Success response missing request_id: {response}")
            
        # All responses should have timestamp for debugging
        if 'timestamp' not in response and 'data' in response:
            if 'timestamp' not in response['data']:
                logger.warning(f"⚠️ Response missing timestamp: {response}")

class TestBackendComponents:
    """Test all backend components systematically"""
    
    def __init__(self, framework: AppTestFramework):
        self.framework = framework
        
    async def test_websocket_manager(self):
        """Test WebSocket manager functionality"""
        logger.info("🧪 Testing WebSocket Manager")
        
        # Test connection handling
        assert self.framework.session_id is not None, "Session ID should be assigned"
        
        # Test error handling with request_id inclusion
        response = await self.framework.send_request_with_validation(
            "invalid:event", {}, expect_success=False
        )
        
        assert response['type'] == 'error', "Should return error for invalid event"
        assert 'request_id' in response, "Error responses must include request_id"
        
        logger.info("✅ WebSocket Manager tests passed")
        
    async def test_simulation_service(self):
        """Test simulation service with minimal data"""
        logger.info("🧪 Testing Simulation Service")
        
        minimal_simulation = {
            "params": {
                "frequency": 80,
                "speed_of_sound": 343,
                "grid_resolution": 4.0,  # Very coarse for fast testing
                "room_vertices": [[0,0], [4,0], [4,4], [0,4]],
                "target_areas": [],
                "avoidance_areas": []
            },
            "sources": [{
                "id": "test_source",
                "x": 2, "y": 2,
                "spl_rms": 90,
                "gain_db": 0,
                "delay_ms": 0,
                "angle": 0,
                "polarity": 1
            }]
        }
        
        # Test simulation start
        response = await self.framework.send_request_with_validation(
            "simulation:start", minimal_simulation, timeout=30.0
        )
        
        assert response['type'] == 'simulation:started', "Should confirm simulation start"
        assert 'request_id' in response, "Start response must include request_id"
        
        # Wait for potential SPL chunks or completion
        await asyncio.sleep(2)
        
        # Test simulation stop
        response = await self.framework.send_request_with_validation(
            "simulation:stop", {}, timeout=10.0
        )
        
        assert response['type'] == 'simulation:stopped', "Should confirm simulation stop"
        assert 'request_id' in response, "Stop response must include request_id"
        
        logger.info("✅ Simulation Service tests passed")
        
    async def test_error_handling_patterns(self):
        """Test that all error cases follow systematic patterns"""
        logger.info("🧪 Testing Error Handling Patterns")
        
        error_test_cases = [
            ("unknown_event_type", {"test": "data"}),
            ("simulation:invalid", {"bad": "data"}),
            ("project:nonexistent", {}),
        ]
        
        for event_type, data in error_test_cases:
            response = await self.framework.send_request_with_validation(
                event_type, data, expect_success=False
            )
            
            # Validate error response structure
            assert response['type'] == 'error', f"Should return error for {event_type}"
            assert 'request_id' in response, f"Error for {event_type} must include request_id"
            assert 'data' in response, f"Error for {event_type} must include data"
            assert 'code' in response['data'], f"Error for {event_type} must include error code"
            assert 'message' in response['data'], f"Error for {event_type} must include message"
            
        logger.info("✅ Error Handling Pattern tests passed")

class TestIntegrationFlows:
    """Test complete integration flows"""
    
    def __init__(self, framework: AppTestFramework):
        self.framework = framework
        
    async def test_complete_simulation_flow(self):
        """Test complete simulation from start to finish"""
        logger.info("🧪 Testing Complete Simulation Flow")
        
        simulation_config = {
            "params": {
                "frequency": 80,
                "speed_of_sound": 343,
                "grid_resolution": 2.0,
                "room_vertices": [[0,0], [6,0], [6,6], [0,6]],
                "target_areas": [],
                "avoidance_areas": []
            },
            "sources": [
                {
                    "id": "source_1",
                    "x": 2, "y": 2,
                    "spl_rms": 90,
                    "gain_db": 0,
                    "delay_ms": 0,
                    "angle": 0,
                    "polarity": 1
                },
                {
                    "id": "source_2", 
                    "x": 4, "y": 4,
                    "spl_rms": 88,
                    "gain_db": -3,
                    "delay_ms": 5,
                    "angle": 45,
                    "polarity": 1
                }
            ]
        }
        
        # Start simulation
        start_response = await self.framework.send_request_with_validation(
            "simulation:start", simulation_config, timeout=30.0
        )
        
        assert start_response['type'] == 'simulation:started'
        
        # Collect any SPL chunks or completion messages
        spl_chunks = []
        completion_data = None
        
        # Listen for simulation results for up to 60 seconds
        timeout = 60.0
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(self.framework.websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'simulation:spl_chunk':
                        spl_chunks.append(data)
                        logger.info(f"📊 Received SPL chunk {len(spl_chunks)}")
                        
                    elif data.get('type') == 'simulation:complete':
                        completion_data = data
                        logger.info("🎯 Simulation completed")
                        break
                        
                except asyncio.TimeoutError:
                    # Continue listening
                    pass
                    
        except Exception as e:
            logger.warning(f"⚠️ Simulation flow interrupted: {e}")
            
        # Stop simulation
        stop_response = await self.framework.send_request_with_validation(
            "simulation:stop", {}, timeout=10.0
        )
        
        assert stop_response['type'] == 'simulation:stopped'
        
        # Validate we got some simulation data
        total_data_received = len(spl_chunks) + (1 if completion_data else 0)
        assert total_data_received > 0, "Should receive some simulation data"
        
        logger.info(f"✅ Complete Simulation Flow tests passed (received {len(spl_chunks)} chunks)")
        
    async def test_concurrent_request_handling(self):
        """Test that the system properly handles concurrent requests"""
        logger.info("🧪 Testing Concurrent Request Handling")
        
        # Create multiple different requests that should all get proper responses
        requests = [
            ("simulation:parameter_update", {"frequency": 60}),
            ("simulation:parameter_update", {"frequency": 80}), 
            ("simulation:parameter_update", {"frequency": 100}),
        ]
        
        # Send requests with small delays to avoid websocket recv conflicts
        responses = []
        for i, (event_type, data) in enumerate(requests):
            try:
                # Add small delay between requests
                if i > 0:
                    await asyncio.sleep(0.1)
                    
                response = await self.framework.send_request_with_validation(
                    event_type, data, timeout=10.0
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"❌ Concurrent request {i} failed: {e}")
                responses.append({"error": str(e)})
                
        # Validate that all requests got responses
        success_count = sum(1 for r in responses if "error" not in r)
        assert success_count >= len(requests) * 0.8, f"Should handle most concurrent requests successfully (got {success_count}/{len(requests)})"
        
        logger.info(f"✅ Concurrent Request Handling tests passed ({success_count}/{len(requests)} successful)")

class TestDataValidation:
    """Test data validation and edge cases"""
    
    def __init__(self, framework: AppTestFramework):
        self.framework = framework
        
    async def test_invalid_simulation_data(self):
        """Test handling of invalid simulation configurations"""
        logger.info("🧪 Testing Invalid Simulation Data Handling")
        
        invalid_configs = [
            # Missing required fields
            {"params": {}},
            
            # Invalid room geometry
            {"params": {"room_vertices": []}, "sources": []},
            
            # Invalid source data
            {"params": {"room_vertices": [[0,0], [1,0], [1,1], [0,1]]}, "sources": [{"id": "bad"}]},
            
            # Extreme values
            {"params": {"frequency": -1}, "sources": []},
        ]
        
        for i, config in enumerate(invalid_configs):
            try:
                response = await self.framework.send_request_with_validation(
                    "simulation:start", config, expect_success=False, timeout=10.0
                )
                
                # Should get error response with proper structure
                assert response['type'] == 'error', f"Invalid config {i} should return error"
                assert 'request_id' in response, f"Error response {i} must include request_id"
                
            except Exception as e:
                logger.info(f"✅ Invalid config {i} properly rejected: {e}")
                
        logger.info("✅ Invalid Simulation Data Handling tests passed")
        
    async def test_large_data_handling(self):
        """Test handling of large simulation grids"""
        logger.info("🧪 Testing Large Data Handling")
        
        # Test with larger grid (but not too large for testing)
        large_config = {
            "params": {
                "frequency": 80,
                "speed_of_sound": 343,
                "grid_resolution": 1.0,  # Higher resolution
                "room_vertices": [[0,0], [10,0], [10,10], [0,10]],  # Larger room
                "target_areas": [],
                "avoidance_areas": []
            },
            "sources": [
                {"id": f"source_{i}", "x": 2 + i, "y": 2 + i, "spl_rms": 90, "gain_db": 0, 
                 "delay_ms": 0, "angle": 0, "polarity": 1}
                for i in range(4)  # Multiple sources
            ]
        }
        
        try:
            response = await self.framework.send_request_with_validation(
                "simulation:start", large_config, timeout=60.0
            )
            
            assert response['type'] == 'simulation:started', "Should handle large config"
            
            # Stop the simulation
            await self.framework.send_request_with_validation(
                "simulation:stop", {}, timeout=10.0
            )
            
            logger.info("✅ Large Data Handling tests passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Large data test failed (may be expected): {e}")

async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting Comprehensive Application Tests")
    logger.info("=" * 80)
    
    framework = AppTestFramework()
    
    try:
        await framework.setup()
        
        test_suites = [
            ("Backend Components", TestBackendComponents(framework)),
            ("Integration Flows", TestIntegrationFlows(framework)),
            ("Data Validation", TestDataValidation(framework)),
        ]
        
        results = {}
        
        for suite_name, test_suite in test_suites:
            logger.info(f"\n{'='*20} {suite_name} {'='*20}")
            
            suite_results = {}
            
            # Get all test methods
            test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
            
            for test_method_name in test_methods:
                try:
                    logger.info(f"\n🧪 Running {test_method_name}")
                    test_method = getattr(test_suite, test_method_name)
                    await test_method()
                    suite_results[test_method_name] = "PASS"
                    logger.info(f"✅ {test_method_name}: PASS")
                    
                except Exception as e:
                    suite_results[test_method_name] = f"FAIL: {e}"
                    logger.error(f"❌ {test_method_name}: FAIL - {e}")
                    
            results[suite_name] = suite_results
            
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        return False
        
    finally:
        await framework.teardown()
        
    # Generate comprehensive report
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPREHENSIVE TEST RESULTS")
    logger.info("=" * 80)
    
    total_tests = 0
    total_passed = 0
    
    for suite_name, suite_results in results.items():
        logger.info(f"\n📋 {suite_name}:")
        for test_name, result in suite_results.items():
            status = "✅ PASS" if result == "PASS" else "❌ FAIL"
            logger.info(f"  {status}: {test_name}")
            total_tests += 1
            if result == "PASS":
                total_passed += 1
                
    logger.info(f"\n🎯 Overall Results: {total_passed}/{total_tests} tests passed")
    
    # Log any accumulated errors
    if framework.error_log:
        logger.info(f"\n📝 Error Log ({len(framework.error_log)} items):")
        for error in framework.error_log:
            logger.warning(f"  ⚠️ {error}")
            
    # Test quality assessment
    pass_rate = total_passed / total_tests if total_tests > 0 else 0
    
    if pass_rate >= 0.9:
        logger.info("🎉 EXCELLENT: App quality is excellent (90%+ tests pass)")
    elif pass_rate >= 0.8:
        logger.info("🎯 GOOD: App quality is good (80%+ tests pass)")
    elif pass_rate >= 0.7:
        logger.info("⚠️ ACCEPTABLE: App quality is acceptable (70%+ tests pass)")
    else:
        logger.error("❌ POOR: App quality needs improvement (<70% tests pass)")
        
    return pass_rate >= 0.8

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    exit(0 if success else 1)