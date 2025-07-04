#!/usr/bin/env python3
"""
Integration Tests for Subwoofer Simulation System
Tests backend-frontend WebSocket communication and simulation execution
"""

import asyncio
import json
import pytest
import websockets
from typing import Dict, Any, List
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from core.config import SimulationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSimulationConfig:
    """Test SimulationConfig class to understand its constructor requirements"""
    
    def test_simulation_config_empty_constructor(self):
        """Test SimulationConfig with empty constructor"""
        try:
            config = SimulationConfig()
            logger.info(f"✅ SimulationConfig() succeeded")
            logger.info(f"Config attributes: {dir(config)}")
            return True
        except Exception as e:
            logger.error(f"❌ SimulationConfig() failed: {e}")
            return False
    
    def test_simulation_config_with_parameters(self):
        """Test SimulationConfig with various parameter combinations"""
        test_cases = [
            # Test case 1: Empty dict
            {},
            # Test case 2: Common parameters
            {
                'room_dimensions': [10, 8, 3],
                'grid_resolution': 0.2,
            },
            # Test case 3: Parameters currently being sent by frontend
            {
                'frequency': 80,
                'speed_of_sound': 343,
                'grid_resolution': 0.2,
                'room_vertices': [[0,0], [10,0], [10,8], [0,8]],
                'target_areas': [],
                'avoidance_areas': []
            }
        ]
        
        results = []
        for i, params in enumerate(test_cases):
            try:
                config = SimulationConfig(**params)
                logger.info(f"✅ Test case {i+1} succeeded with params: {params}")
                results.append(True)
            except Exception as e:
                logger.error(f"❌ Test case {i+1} failed with params {params}: {e}")
                results.append(False)
        
        return results
    
    def test_simulation_config_inspection(self):
        """Inspect SimulationConfig to understand expected parameters"""
        try:
            import inspect
            
            # Get the constructor signature
            sig = inspect.signature(SimulationConfig.__init__)
            logger.info(f"SimulationConfig.__init__ signature: {sig}")
            
            # Get all parameters
            params = list(sig.parameters.keys())
            logger.info(f"Constructor parameters: {params}")
            
            # Try to create with default values
            config = SimulationConfig()
            
            # Inspect attributes
            attrs = [attr for attr in dir(config) if not attr.startswith('_')]
            logger.info(f"Config attributes: {attrs}")
            
            # Get actual values
            for attr in attrs[:10]:  # First 10 to avoid spam
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        logger.info(f"  {attr}: {value}")
                except:
                    pass
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Config inspection failed: {e}")
            return False

class TestWebSocketCommunication:
    """Test WebSocket communication between frontend and backend"""
    
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        try:
            uri = "ws://localhost:8001/ws"
            async with websockets.connect(uri) as websocket:
                logger.info("✅ WebSocket connection established")
                
                # Wait for welcome message
                welcome_msg = await websocket.recv()
                welcome_data = json.loads(welcome_msg)
                logger.info(f"📨 Welcome message: {welcome_data}")
                
                assert welcome_data['type'] == 'connected'
                assert 'sessionId' in welcome_data['data']
                
                return True
                
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def test_simulation_start_message_format(self):
        """Test the exact message format being sent by frontend"""
        frontend_message = {
            "type": "simulation:start",
            "data": {
                "params": {
                    "frequency": 80,
                    "speed_of_sound": 343,
                    "grid_resolution": 0.2,
                    "room_vertices": [[0,0], [10,0], [10,8], [0,8]],
                    "target_areas": [],
                    "avoidance_areas": []
                },
                "sources": [
                    {
                        "id": "source_1",
                        "x": 2,
                        "y": 4,
                        "spl_rms": 105,
                        "gain_db": 0,
                        "delay_ms": 0,
                        "angle": 0,
                        "polarity": 1
                    }
                ]
            }
        }
        
        try:
            uri = "ws://localhost:8001/ws"
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await websocket.recv()
                
                # Send simulation start
                await websocket.send(json.dumps(frontend_message))
                logger.info("📤 Sent simulation start message")
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                logger.info(f"📨 Response: {response_data}")
                
                return response_data
                
        except Exception as e:
            logger.error(f"❌ Simulation start test failed: {e}")
            return None

class TestParameterMapping:
    """Test parameter mapping between frontend and backend"""
    
    def test_parameter_conversion(self):
        """Test converting frontend parameters to backend format"""
        frontend_params = {
            "frequency": 80,
            "speed_of_sound": 343,
            "grid_resolution": 0.2,
            "room_vertices": [[0,0], [10,0], [10,8], [0,8]],
            "target_areas": [],
            "avoidance_areas": []
        }
        
        # Test what parameters SimulationConfig actually accepts
        try:
            # Try the direct approach first
            config = SimulationConfig(**frontend_params)
            logger.info("✅ Direct parameter mapping worked")
            return True
        except Exception as e:
            logger.error(f"❌ Direct mapping failed: {e}")
            
            # Try without problematic parameters
            safe_params = {k: v for k, v in frontend_params.items() 
                          if k not in ['frequency', 'speed_of_sound']}
            try:
                config = SimulationConfig(**safe_params)
                logger.info(f"✅ Safe parameter mapping worked with: {list(safe_params.keys())}")
                return True
            except Exception as e2:
                logger.error(f"❌ Safe mapping also failed: {e2}")
                return False

async def run_integration_tests():
    """Run all integration tests"""
    logger.info("🧪 Starting Integration Tests")
    logger.info("=" * 50)
    
    # Test 1: SimulationConfig
    logger.info("\n📋 Test 1: SimulationConfig Analysis")
    config_test = TestSimulationConfig()
    
    logger.info("1.1 Empty constructor test:")
    config_test.test_simulation_config_empty_constructor()
    
    logger.info("\n1.2 Parameter combinations test:")
    results = config_test.test_simulation_config_with_parameters()
    logger.info(f"Results: {results}")
    
    logger.info("\n1.3 Config inspection:")
    config_test.test_simulation_config_inspection()
    
    # Test 2: Parameter Mapping
    logger.info("\n📋 Test 2: Parameter Mapping")
    param_test = TestParameterMapping()
    param_test.test_parameter_conversion()
    
    # Test 3: WebSocket Communication
    logger.info("\n📋 Test 3: WebSocket Communication")
    ws_test = TestWebSocketCommunication()
    
    logger.info("3.1 Basic connection test:")
    try:
        await ws_test.test_websocket_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
    
    logger.info("\n3.2 Simulation start message test:")
    try:
        response = await ws_test.test_simulation_start_message_format()
        if response:
            logger.info("✅ Message format test completed")
    except Exception as e:
        logger.error(f"Message test failed: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("🏁 Integration tests completed")

if __name__ == "__main__":
    asyncio.run(run_integration_tests())