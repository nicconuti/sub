#!/usr/bin/env python3
"""
Unit Tests for WebSocket Timeout Issues
Systematic debugging of request-response flow
"""

import asyncio
import json
import pytest
import websockets
import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketTestClient:
    def __init__(self, uri: str = "ws://localhost:8001/ws"):
        self.uri = uri
        self.websocket = None
        self.session_id = None
        self.received_messages = []
        self.pending_requests = {}
        
    async def connect(self):
        """Test WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.uri, max_size=16777216)
            
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get('type') == 'connected':
                self.session_id = welcome_data['data']['sessionId']
                logger.info(f"✅ Connected with session: {self.session_id}")
                return True
            else:
                logger.error(f"❌ Unexpected welcome message: {welcome_data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_request(self, event_type: str, data: Dict[str, Any], timeout: float = 10.0):
        """Send request and wait for response with timeout tracking"""
        if not self.websocket:
            raise Exception("Not connected")
            
        request_id = f"test_req_{int(time.time() * 1000)}_{event_type}"
        
        message = {
            "type": event_type,
            "data": data,
            "request_id": request_id,
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        logger.info(f"📤 Sending {event_type} with request_id: {request_id}")
        
        # Track the request
        start_time = time.time()
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"✅ Message sent successfully")
            
            # Wait for response
            while True:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
                    response_data = json.loads(response)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"📥 Received: {response_data.get('type')} (elapsed: {elapsed:.2f}s)")
                    
                    # Check if this is our response
                    if response_data.get('request_id') == request_id:
                        logger.info(f"✅ Got matching response for {request_id} in {elapsed:.2f}s")
                        return response_data
                    else:
                        # Store non-matching messages
                        self.received_messages.append(response_data)
                        logger.info(f"📨 Stored non-matching message: {response_data.get('type')}")
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.error(f"❌ Request {request_id} timed out after {elapsed:.2f}s")
                    raise Exception(f"Request timeout: {event_type}")
                    
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Request failed after {elapsed:.2f}s: {e}")
            raise
    
    async def disconnect(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

class TestWebSocketTimeout:
    """Systematic tests for WebSocket timeout issues"""
    
    def __init__(self):
        self.client = WebSocketTestClient()
        
    async def test_basic_connection(self):
        """Test 1: Basic WebSocket connection"""
        logger.info("\n🧪 Test 1: Basic Connection")
        
        success = await self.client.connect()
        assert success, "Basic connection should work"
        
        await self.client.disconnect()
        logger.info("✅ Test 1 passed")
        
    async def test_simple_request_response(self):
        """Test 2: Simple request-response without simulation"""
        logger.info("\n🧪 Test 2: Simple Request-Response")
        
        await self.client.connect()
        
        # Send a simple ping-like request
        try:
            response = await self.client.send_request("client:ping", {"timestamp": time.time()}, timeout=5.0)
            logger.info(f"✅ Got response: {response}")
        except Exception as e:
            logger.error(f"❌ Simple request failed: {e}")
            raise
        finally:
            await self.client.disconnect()
            
        logger.info("✅ Test 2 passed")
        
    async def test_simulation_start_minimal(self):
        """Test 3: Minimal simulation start request"""
        logger.info("\n🧪 Test 3: Minimal Simulation Start")
        
        await self.client.connect()
        
        minimal_simulation_data = {
            "params": {
                "frequency": 80,
                "speed_of_sound": 343,
                "grid_resolution": 2.0,  # Very coarse grid
                "room_vertices": [[0,0], [2,0], [2,2], [0,2]],  # Tiny room
                "target_areas": [],
                "avoidance_areas": []
            },
            "sources": [
                {
                    "id": "test_source",
                    "x": 1,
                    "y": 1,
                    "spl_rms": 90,
                    "gain_db": 0,
                    "delay_ms": 0,
                    "angle": 0,
                    "polarity": 1
                }
            ]
        }
        
        try:
            response = await self.client.send_request("simulation:start", minimal_simulation_data, timeout=15.0)
            logger.info(f"✅ Simulation start response: {response}")
        except Exception as e:
            logger.error(f"❌ Simulation start failed: {e}")
            raise
        finally:
            await self.client.disconnect()
            
        logger.info("✅ Test 3 passed")
        
    async def test_backend_session_tracking(self):
        """Test 4: Backend session state tracking"""
        logger.info("\n🧪 Test 4: Backend Session Tracking")
        
        await self.client.connect()
        
        # Check if session is properly tracked
        try:
            # Send multiple requests to test session persistence
            for i in range(3):
                response = await self.client.send_request("client:ping", {"test": f"ping_{i}"}, timeout=5.0)
                logger.info(f"Ping {i}: {response}")
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Session tracking test failed: {e}")
            raise
        finally:
            await self.client.disconnect()
            
        logger.info("✅ Test 4 passed")
        
    async def test_concurrent_requests(self):
        """Test 5: Concurrent request handling"""
        logger.info("\n🧪 Test 5: Concurrent Requests")
        
        await self.client.connect()
        
        try:
            # Send multiple requests concurrently
            tasks = []
            for i in range(3):
                task = self.client.send_request("client:ping", {"concurrent": f"test_{i}"}, timeout=5.0)
                tasks.append(task)
                
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"❌ Concurrent request {i} failed: {response}")
                    raise response
                else:
                    logger.info(f"✅ Concurrent request {i} succeeded: {response}")
                    
        except Exception as e:
            logger.error(f"❌ Concurrent test failed: {e}")
            raise
        finally:
            await self.client.disconnect()
            
        logger.info("✅ Test 5 passed")

async def run_systematic_tests():
    """Run all tests systematically"""
    logger.info("🚀 Starting Systematic WebSocket Timeout Tests")
    logger.info("=" * 60)
    
    tester = TestWebSocketTimeout()
    tests = [
        ("Basic Connection", tester.test_basic_connection),
        ("Simple Request-Response", tester.test_simple_request_response),
        ("Minimal Simulation Start", tester.test_simulation_start_minimal),
        ("Session Tracking", tester.test_backend_session_tracking),
        ("Concurrent Requests", tester.test_concurrent_requests)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            await test_func()
            results[test_name] = "PASS"
            logger.info(f"✅ {test_name}: PASS")
        except Exception as e:
            results[test_name] = f"FAIL: {e}"
            logger.error(f"❌ {test_name}: FAIL - {e}")
            
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASS" else f"❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result == "PASS":
            passed += 1
            
    logger.info(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed < len(tests):
        logger.error("❌ Some tests failed - need investigation")
        return False
    else:
        logger.info("✅ All tests passed")
        return True

if __name__ == "__main__":
    asyncio.run(run_systematic_tests())