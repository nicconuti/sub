"""
Simulation API Routes
REST endpoints for SPL simulation management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
import time

from ..models.events import SimulationParams, SourceData
from ..services.simulation_service import SimulationService
from ..services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection (will be set by main.py)
_simulation_service: Optional[SimulationService] = None
_websocket_manager: Optional[WebSocketManager] = None

def get_simulation_service() -> SimulationService:
    """Get simulation service instance"""
    if _simulation_service is None:
        raise HTTPException(status_code=500, detail="Simulation service not initialized")
    return _simulation_service

def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance"""
    if _websocket_manager is None:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    return _websocket_manager

def set_dependencies(simulation_service: SimulationService, websocket_manager: WebSocketManager):
    """Set service dependencies (called from main.py)"""
    global _simulation_service, _websocket_manager
    _simulation_service = simulation_service
    _websocket_manager = websocket_manager

@router.post("/calculate")
async def calculate_spl(
    params: SimulationParams,
    sources: List[SourceData],
    background_tasks: BackgroundTasks,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Calculate SPL map (synchronous version for REST API)
    
    For real-time streaming, use WebSocket connection instead
    """
    try:
        # Validate input
        if not sources:
            raise HTTPException(status_code=400, detail="At least one source required")
            
        # This is a simplified synchronous version
        # Real implementations should use WebSocket for streaming
        
        # For now, return basic response
        return {
            "message": "SPL calculation started",
            "recommendation": "Use WebSocket connection for real-time streaming results",
            "websocket_url": "/ws",
            "sources_count": len(sources),
            "grid_size": _estimate_grid_size(params),
            "estimated_time": _estimate_computation_time(len(sources), params)
        }
        
    except Exception as e:
        logger.error(f"❌ SPL calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.post("/validate")
async def validate_simulation(
    params: SimulationParams,
    sources: List[SourceData]
):
    """
    Validate simulation parameters without running calculation
    """
    try:
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Validate parameters
        if params.frequency < 20 or params.frequency > 20000:
            validation_results["errors"].append("Frequency must be between 20-20000 Hz")
            validation_results["valid"] = False
            
        if params.grid_resolution < 0.01 or params.grid_resolution > 1.0:
            validation_results["errors"].append("Grid resolution must be between 0.01-1.0 m")
            validation_results["valid"] = False
            
        if params.speed_of_sound < 300 or params.speed_of_sound > 400:
            validation_results["errors"].append("Speed of sound must be between 300-400 m/s")
            validation_results["valid"] = False
            
        # Validate sources
        if not sources:
            validation_results["errors"].append("At least one source required")
            validation_results["valid"] = False
            
        for i, source in enumerate(sources):
            if source.spl_rms < 50 or source.spl_rms > 150:
                validation_results["warnings"].append(f"Source {i+1}: SPL RMS outside typical range (50-150 dB)")
                
            if source.gain_db < -60 or source.gain_db > 20:
                validation_results["warnings"].append(f"Source {i+1}: Gain outside typical range (-60 to +20 dB)")
                
        # Performance recommendations
        grid_size = _estimate_grid_size(params)
        if grid_size > 100000:
            validation_results["recommendations"].append(
                "Large grid size detected. Consider increasing grid_resolution for faster calculation."
            )
            
        if len(sources) > 50:
            validation_results["recommendations"].append(
                "Many sources detected. Consider grouping nearby sources for better performance."
            )
            
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/status/{simulation_id}")
async def get_simulation_status(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """Get status of specific simulation"""
    try:
        status = simulation_service.get_simulation_status(simulation_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting simulation status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/cancel/{simulation_id}")
async def cancel_simulation(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """Cancel specific simulation"""
    try:
        # Note: This requires session_id, but we don't have it in REST context
        # This endpoint is mainly for administrative purposes
        
        status = simulation_service.get_simulation_status(simulation_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        # Cancel via WebSocket manager (requires session_id)
        session_id = status["session_id"]
        success = await simulation_service.stop_simulation(session_id, simulation_id)
        
        if success:
            return {"message": f"Simulation {simulation_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel simulation")
            
    except Exception as e:
        logger.error(f"❌ Error cancelling simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

@router.get("/stats")
async def get_simulation_stats(
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """Get simulation service statistics"""
    try:
        stats = simulation_service.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.get("/presets")
async def get_simulation_presets():
    """Get predefined simulation presets"""
    try:
        presets = {
            "small_room": {
                "name": "Small Room (5x5m)",
                "params": {
                    "frequency": 80.0,
                    "speed_of_sound": 343.0,
                    "grid_resolution": 0.1,
                    "room_vertices": [[0, 0], [5, 0], [5, 5], [0, 5]]
                },
                "sources": [
                    {
                        "id": "sub1",
                        "x": 2.5,
                        "y": 1.0,
                        "spl_rms": 105.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    }
                ]
            },
            "medium_room": {
                "name": "Medium Room (10x8m)",
                "params": {
                    "frequency": 60.0,
                    "speed_of_sound": 343.0,
                    "grid_resolution": 0.15,
                    "room_vertices": [[0, 0], [10, 0], [10, 8], [0, 8]]
                },
                "sources": [
                    {
                        "id": "sub1",
                        "x": 2.0,
                        "y": 1.5,
                        "spl_rms": 110.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    },
                    {
                        "id": "sub2",
                        "x": 8.0,
                        "y": 1.5,
                        "spl_rms": 110.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    }
                ]
            },
            "large_venue": {
                "name": "Large Venue (20x15m)",
                "params": {
                    "frequency": 40.0,
                    "speed_of_sound": 343.0,
                    "grid_resolution": 0.2,
                    "room_vertices": [[0, 0], [20, 0], [20, 15], [0, 15]]
                },
                "sources": [
                    {
                        "id": "sub1",
                        "x": 5.0,
                        "y": 2.0,
                        "spl_rms": 115.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    },
                    {
                        "id": "sub2",
                        "x": 10.0,
                        "y": 2.0,
                        "spl_rms": 115.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    },
                    {
                        "id": "sub3",
                        "x": 15.0,
                        "y": 2.0,
                        "spl_rms": 115.0,
                        "gain_db": 0.0,
                        "delay_ms": 0.0,
                        "angle": 0.0,
                        "polarity": 1
                    }
                ]
            }
        }
        
        return presets
        
    except Exception as e:
        logger.error(f"❌ Error getting presets: {e}")
        raise HTTPException(status_code=500, detail=f"Presets retrieval failed: {str(e)}")

# Helper functions
def _estimate_grid_size(params: SimulationParams) -> int:
    """Estimate grid size from parameters"""
    try:
        if not params.room_vertices:
            return 10000  # Default estimate
            
        # Calculate room bounds
        xs = [v[0] for v in params.room_vertices]
        ys = [v[1] for v in params.room_vertices]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        points_x = int(width / params.grid_resolution)
        points_y = int(height / params.grid_resolution)
        
        return points_x * points_y
        
    except Exception:
        return 10000  # Fallback estimate

def _estimate_computation_time(source_count: int, params: SimulationParams) -> float:
    """Estimate computation time"""
    try:
        grid_size = _estimate_grid_size(params)
        # Rough estimation based on grid size and source count
        base_time = (grid_size * source_count) / 500000  # seconds
        return max(1.0, base_time)
        
    except Exception:
        return 5.0  # Fallback estimate