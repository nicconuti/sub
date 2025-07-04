#!/usr/bin/env python3
"""
Test validation error request_id inclusion
"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_validation_error():
    """Test that validation errors include request_id"""
    
    try:
        uri = "ws://localhost:8001/ws"
        async with websockets.connect(uri) as websocket:
            # Get welcome
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            session_id = welcome_data['data']['sessionId']
            logger.info(f"Connected: {session_id}")
            
            # Send simulation start with invalid grid_resolution (too high)
            test_message = {
                "type": "simulation:start",
                "data": {
                    "params": {
                        "frequency": 80,
                        "speed_of_sound": 343,
                        "grid_resolution": 4.0,  # This should trigger validation error
                        "room_vertices": [[0,0], [4,0], [4,4], [0,4]]
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
                },
                "request_id": "validation_test_123",
                "session_id": session_id,
                "timestamp": 1234567890
            }
            
            await websocket.send(json.dumps(test_message))
            logger.info("Sent simulation start with invalid params")
            
            # Get the error response
            error_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            error_data = json.loads(error_response)
            
            logger.info("VALIDATION ERROR RESPONSE:")
            logger.info(f"  Type: {error_data.get('type')}")
            logger.info(f"  Data: {error_data.get('data')}")
            logger.info(f"  Request ID: {error_data.get('request_id')}")
            logger.info(f"  Full: {error_data}")
            
            # Verify request_id is included
            if error_data.get('request_id') == 'validation_test_123':
                logger.info("✅ SUCCESS: Validation error includes request_id")
                return True
            else:
                logger.error("❌ FAILURE: Validation error missing request_id")
                return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_validation_error())
    exit(0 if success else 1)