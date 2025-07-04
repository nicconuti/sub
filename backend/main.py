"""
FastAPI WebSocket Server per Subwoofer Simulation
Ottimizzato per comunicazione real-time e performance elevate
"""

import asyncio
import logging
from typing import Dict, List, Optional
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.services.websocket_manager import WebSocketManager
from api.services.simulation_service import SimulationService
from api.models.events import ClientEvent, ServerEvent
from api.routes import simulation, projects, optimization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
websocket_manager = WebSocketManager()
simulation_service = SimulationService(websocket_manager)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("🚀 Starting Subwoofer Simulation API Server")
    
    # Startup
    await websocket_manager.startup()
    await simulation_service.startup()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server")
    await websocket_manager.shutdown()
    await simulation_service.shutdown()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Subwoofer Simulation API",
    description="Real-time acoustic simulation with WebSocket communication",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up route dependencies
simulation.set_dependencies(simulation_service, websocket_manager)
optimization.set_websocket_manager(websocket_manager)

# Include API routes
app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Subwoofer Simulation API", 
        "version": "2.0.0",
        "status": "running",
        "websocket_connections": len(websocket_manager.active_connections)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "websocket_manager": {
            "active_connections": len(websocket_manager.active_connections),
            "total_sessions": websocket_manager.session_count
        },
        "simulation_service": {
            "active_simulations": len(simulation_service.active_simulations),
            "completed_simulations": simulation_service.completed_count
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint per comunicazione real-time
    
    Gestisce:
    - Connection management
    - Event routing
    - Error handling
    - Session management
    """
    session_id = await websocket_manager.connect(websocket)
    logger.info(f"🔗 WebSocket connected: session {session_id}")
    
    try:
        # Send welcome message
        await websocket_manager.send_to_session(session_id, {
            "type": "connected",
            "data": {
                "sessionId": session_id,
                "timestamp": asyncio.get_event_loop().time(),
                "serverVersion": "2.0.0"
            }
        })
        
        # Event loop per incoming messages
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=300.0  # 5 minute timeout
                )
                
                logger.info(f"📥 Received message from {session_id}: {message[:200]}...")
                
                # Parse event
                try:
                    event_data = json.loads(message)
                    event = ClientEvent.parse_obj(event_data)
                    logger.info(f"📋 Parsed event: {event.type} (request_id: {event.request_id})")
                except Exception as e:
                    logger.error(f"❌ Invalid event format from {session_id}: {e}")
                    logger.error(f"📄 Raw message: {message}")
                    
                    # Try to extract request_id from raw message for proper error matching
                    request_id = None
                    try:
                        if isinstance(event_data, dict):
                            request_id = event_data.get('request_id')
                    except:
                        pass
                    
                    await websocket_manager.send_error(session_id, "invalid_event", str(e), request_id)
                    continue
                
                # Route event to appropriate handler
                await route_client_event(session_id, event)
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout for session {session_id}")
                break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"❌ Error processing message from {session_id}: {e}")
                await websocket_manager.send_error(session_id, "processing_error", str(e), None)
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"❌ WebSocket error for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(session_id)
        logger.info(f"🔌 WebSocket disconnected: session {session_id}")

async def route_client_event(session_id: str, event: ClientEvent):
    """
    Route client events to appropriate handlers
    """
    try:
        if event.type.startswith("simulation:"):
            await handle_simulation_event(session_id, event)
        elif event.type.startswith("project:"):
            await handle_project_event(session_id, event)
        elif event.type.startswith("optimization:"):
            await handle_optimization_event(session_id, event)
        elif event.type.startswith("source:") or event.type.startswith("room:"):
            await handle_interactive_event(session_id, event)
        else:
            logger.warning(f"⚠️ Unknown event type: {event.type}")
            await websocket_manager.send_error(session_id, "unknown_event", f"Unknown event type: {event.type}", event.request_id)
            
    except Exception as e:
        logger.error(f"❌ Error routing event {event.type} from {session_id}: {e}")
        await websocket_manager.send_error(session_id, "routing_error", str(e), getattr(event, 'request_id', None))

async def handle_simulation_event(session_id: str, event: ClientEvent):
    """Handle simulation-related events"""
    logger.info(f"🧮 Handling simulation event: {event.type} from {session_id}")
    
    try:
        if event.type == "simulation:start":
            logger.info(f"🚀 Starting simulation for session {session_id}")
            await simulation_service.start_simulation(session_id, event.data, event.request_id)
            
            # Send immediate confirmation response
            await websocket_manager.send_to_session(session_id, {
                "type": "simulation:started",
                "data": {"status": "started", "message": "Simulation started successfully"},
                "request_id": event.request_id,
                "timestamp": asyncio.get_event_loop().time()
            })
            
        elif event.type == "simulation:stop":
            logger.info(f"🛑 Stopping simulation for session {session_id}")
            await simulation_service.stop_simulation(session_id, event.request_id)
            
            # Send immediate confirmation response
            await websocket_manager.send_to_session(session_id, {
                "type": "simulation:stopped", 
                "data": {"status": "stopped", "message": "Simulation stopped successfully"},
                "request_id": event.request_id,
                "timestamp": asyncio.get_event_loop().time()
            })
            
        elif event.type == "simulation:parameter_update":
            logger.info(f"⚙️ Updating simulation parameters for session {session_id}")
            await simulation_service.update_parameter(session_id, event.data, event.request_id)
            
            # Send immediate confirmation response
            await websocket_manager.send_to_session(session_id, {
                "type": "simulation:parameter_updated",
                "data": {"status": "updated", "message": "Parameters updated successfully"},
                "request_id": event.request_id,
                "timestamp": asyncio.get_event_loop().time()
            })
            
        else:
            logger.warning(f"⚠️ Unknown simulation event: {event.type}")
            
    except Exception as e:
        logger.error(f"❌ Error handling simulation event {event.type}: {e}")
        # Send error response
        await websocket_manager.send_error(session_id, f"simulation_{event.type}_error", str(e), event.request_id)

async def handle_project_event(session_id: str, event: ClientEvent):
    """Handle project management events"""
    # Implementation will be added in project management phase
    logger.info(f"📁 Project event {event.type} from {session_id}")

async def handle_optimization_event(session_id: str, event: ClientEvent):
    """Handle optimization events"""
    # Implementation will be added in optimization phase
    logger.info(f"🧬 Optimization event {event.type} from {session_id}")

async def handle_interactive_event(session_id: str, event: ClientEvent):
    """Handle real-time interactive events (drag & drop, etc.)"""
    if event.type == "source:move":
        # Broadcast to other sessions in same project
        await websocket_manager.broadcast_to_project(
            event.data.get("projectId"), 
            {
                "type": "ui:source_moved",
                "data": event.data
            },
            exclude_session=session_id
        )
    elif event.type == "source:rotate":
        # Handle source rotation
        await websocket_manager.broadcast_to_project(
            event.data.get("projectId"),
            {
                "type": "ui:source_rotated",
                "data": event.data
            },
            exclude_session=session_id
        )
    elif event.type == "room:vertex_move":
        await websocket_manager.broadcast_to_project(
            event.data.get("projectId"),
            {
                "type": "ui:room_vertex_moved", 
                "data": event.data
            },
            exclude_session=session_id
        )
    elif event.type.startswith("spl_preview:"):
        # Handle SPL preview requests (hover effects)
        await handle_spl_preview_event(session_id, event)

async def handle_spl_preview_event(session_id: str, event: ClientEvent):
    """Handle SPL preview for UI hover effects"""
    try:
        point = event.data.get("point", [0, 0])
        frequency = event.data.get("frequency", 80.0)
        
        # Quick SPL calculation for preview (simplified)
        # In production, this would use cached data or fast approximation
        spl_value = await simulation_service.calculate_point_spl(
            point[0], point[1], frequency
        )
        
        # Send preview response
        await websocket_manager.send_to_session(session_id, {
            "type": "ui:spl_preview",
            "data": {
                "point": point,
                "spl_value": spl_value,
                "frequency": frequency,
                "timestamp": asyncio.get_event_loop().time()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ SPL preview error: {e}")
        await websocket_manager.send_error(session_id, "spl_preview_error", str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": "http_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "server_error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
        ws_max_size=16777216  # 16MB WebSocket message limit
    )