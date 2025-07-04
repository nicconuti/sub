#!/usr/bin/env python3
"""
Debug the specific error messages from WebSocket
"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_error_messages():
    """Check what error messages we're getting"""
    
    try:
        uri = "ws://localhost:8001/ws"
        async with websockets.connect(uri) as websocket:
            # Get welcome
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            session_id = welcome_data['data']['sessionId']
            logger.info(f"Connected: {session_id}")
            
            # Send an unsupported event to see error structure
            test_message = {
                "type": "client:ping",
                "data": {"test": "data"},
                "request_id": "debug_req_123",
                "session_id": session_id,
                "timestamp": 1234567890
            }
            
            await websocket.send(json.dumps(test_message))
            logger.info("Sent test message")
            
            # Get the error response
            error_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            error_data = json.loads(error_response)
            
            logger.info("ERROR RESPONSE STRUCTURE:")
            logger.info(f"  Type: {error_data.get('type')}")
            logger.info(f"  Data: {error_data.get('data')}")
            logger.info(f"  Request ID: {error_data.get('request_id')}")
            logger.info(f"  Full: {error_data}")
            
    except Exception as e:
        logger.error(f"Debug failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_error_messages())