"""
Optimization API Routes
REST endpoints for genetic algorithm optimization
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
import uuid
import time
import asyncio
from datetime import datetime

from ..models.events import OptimizationConfig, OptimizationGeneration, SourceData
from ..services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for optimization sessions
_optimization_sessions: Dict[str, Dict[str, Any]] = {}
_websocket_manager: Optional[WebSocketManager] = None

def get_websocket_manager() -> WebSocketManager:
    """Get WebSocket manager instance"""
    if _websocket_manager is None:
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    return _websocket_manager

def set_websocket_manager(websocket_manager: WebSocketManager):
    """Set WebSocket manager dependency"""
    global _websocket_manager
    _websocket_manager = websocket_manager

@router.post("/start")
async def start_optimization(
    config: OptimizationConfig,
    project_id: str,
    sources: List[SourceData],
    background_tasks: BackgroundTasks,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Start genetic algorithm optimization
    
    For real-time progress updates, use WebSocket connection
    """
    try:
        # Validate input
        if not sources:
            raise HTTPException(status_code=400, detail="At least one source required for optimization")
        
        # Generate optimization session ID
        optimization_id = str(uuid.uuid4())
        
        # Create optimization session
        session = {
            "id": optimization_id,
            "project_id": project_id,
            "config": config.dict(),
            "sources": [source.dict() for source in sources],
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "progress": {
                "current_generation": 0,
                "total_generations": config.max_generations,
                "best_fitness": None,
                "average_fitness": None,
                "convergence_rate": 0.0,
                "estimated_time_remaining": None
            },
            "results": None,
            "error": None
        }
        
        # Store session
        _optimization_sessions[optimization_id] = session
        
        # Start background optimization task
        background_tasks.add_task(
            _run_optimization_simulation,
            optimization_id,
            config,
            sources,
            websocket_manager
        )
        
        logger.info(f"🧬 Started optimization {optimization_id} for project {project_id}")
        
        return {
            "message": "Optimization started successfully",
            "optimization_id": optimization_id,
            "project_id": project_id,
            "config": config.dict(),
            "estimated_duration": _estimate_optimization_time(config),
            "websocket_url": "/ws",
            "recommendation": "Connect to WebSocket for real-time progress updates"
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization start failed: {str(e)}")

@router.get("/{optimization_id}")
async def get_optimization_status(optimization_id: str):
    """Get optimization status and progress"""
    try:
        if optimization_id not in _optimization_sessions:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        session = _optimization_sessions[optimization_id]
        
        return {
            "optimization_id": optimization_id,
            "project_id": session["project_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "progress": session["progress"],
            "results": session["results"],
            "error": session["error"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.delete("/{optimization_id}")
async def cancel_optimization(optimization_id: str):
    """Cancel running optimization"""
    try:
        if optimization_id not in _optimization_sessions:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        session = _optimization_sessions[optimization_id]
        
        if session["status"] == "running":
            session["status"] = "cancelled"
            session["error"] = "Optimization cancelled by user"
            
            logger.info(f"🛑 Cancelled optimization {optimization_id}")
            
            return {
                "message": "Optimization cancelled successfully",
                "optimization_id": optimization_id
            }
        else:
            return {
                "message": f"Optimization already {session['status']}",
                "optimization_id": optimization_id
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error cancelling optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

@router.get("/")
async def list_optimizations():
    """List all optimization sessions"""
    try:
        sessions = list(_optimization_sessions.values())
        
        # Sort by creation date (most recent first)
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        
        # Return summary info only
        summaries = []
        for session in sessions:
            summary = {
                "id": session["id"],
                "project_id": session["project_id"],
                "status": session["status"],
                "created_at": session["created_at"],
                "progress": {
                    "current_generation": session["progress"]["current_generation"],
                    "total_generations": session["progress"]["total_generations"],
                    "completion_percentage": (
                        session["progress"]["current_generation"] / 
                        max(1, session["progress"]["total_generations"])
                    ) * 100
                },
                "sources_count": len(session["sources"]),
                "has_results": session["results"] is not None
            }
            summaries.append(summary)
        
        return summaries
        
    except Exception as e:
        logger.error(f"❌ Error listing optimizations: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@router.get("/{optimization_id}/results")
async def get_optimization_results(optimization_id: str):
    """Get detailed optimization results"""
    try:
        if optimization_id not in _optimization_sessions:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        session = _optimization_sessions[optimization_id]
        
        if session["status"] != "completed":
            raise HTTPException(status_code=400, detail="Optimization not completed yet")
        
        if session["results"] is None:
            raise HTTPException(status_code=404, detail="No results available")
        
        return {
            "optimization_id": optimization_id,
            "project_id": session["project_id"],
            "results": session["results"],
            "performance_metrics": {
                "total_generations": session["progress"]["total_generations"],
                "final_fitness": session["progress"]["best_fitness"],
                "convergence_achieved": session["results"]["convergence_data"][-1] if session["results"]["convergence_data"] else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting optimization results: {e}")
        raise HTTPException(status_code=500, detail=f"Results retrieval failed: {str(e)}")

@router.get("/presets")
async def get_optimization_presets():
    """Get predefined optimization presets"""
    try:
        presets = {
            "quick": {
                "name": "Quick Optimization",
                "description": "Fast optimization for quick results",
                "config": {
                    "population_size": 30,
                    "max_generations": 50,
                    "mutation_rate": 0.02,
                    "crossover_rate": 0.8,
                    "target_spl": 85,
                    "tolerance": 5
                },
                "estimated_time": "2-5 minutes"
            },
            "balanced": {
                "name": "Balanced Optimization",
                "description": "Good balance between speed and quality",
                "config": {
                    "population_size": 50,
                    "max_generations": 100,
                    "mutation_rate": 0.01,
                    "crossover_rate": 0.8,
                    "target_spl": 85,
                    "tolerance": 3
                },
                "estimated_time": "5-15 minutes"
            },
            "thorough": {
                "name": "Thorough Optimization",
                "description": "Comprehensive search for best results",
                "config": {
                    "population_size": 100,
                    "max_generations": 200,
                    "mutation_rate": 0.005,
                    "crossover_rate": 0.9,
                    "target_spl": 85,
                    "tolerance": 2
                },
                "estimated_time": "15-45 minutes"
            }
        }
        
        return presets
        
    except Exception as e:
        logger.error(f"❌ Error getting optimization presets: {e}")
        raise HTTPException(status_code=500, detail=f"Presets retrieval failed: {str(e)}")

@router.get("/stats")
async def get_optimization_stats():
    """Get optimization statistics"""
    try:
        total_optimizations = len(_optimization_sessions)
        
        # Count by status
        status_counts = {}
        for session in _optimization_sessions.values():
            status = session["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Find most successful optimization
        best_optimization = None
        best_fitness = None
        
        for session in _optimization_sessions.values():
            if session["status"] == "completed" and session["results"]:
                fitness = session["progress"]["best_fitness"]
                if fitness is not None and (best_fitness is None or fitness > best_fitness):
                    best_fitness = fitness
                    best_optimization = session
        
        return {
            "total_optimizations": total_optimizations,
            "status_distribution": status_counts,
            "success_rate": status_counts.get("completed", 0) / max(1, total_optimizations) * 100,
            "best_optimization": {
                "id": best_optimization["id"],
                "project_id": best_optimization["project_id"],
                "fitness": best_fitness,
                "generations": best_optimization["progress"]["total_generations"]
            } if best_optimization else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting optimization stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

# Background task functions
async def _run_optimization_simulation(
    optimization_id: str,
    config: OptimizationConfig,
    sources: List[SourceData],
    websocket_manager: WebSocketManager
):
    """
    Run genetic algorithm optimization simulation
    """
    try:
        session = _optimization_sessions[optimization_id]
        
        # Simulate genetic algorithm progression
        for generation in range(config.max_generations):
            # Check if cancelled
            if session["status"] == "cancelled":
                break
            
            # Simulate generation processing
            await asyncio.sleep(0.1)  # Simulate computation time
            
            # Update progress
            progress = session["progress"]
            progress["current_generation"] = generation + 1
            
            # Simulate fitness improvement
            if progress["best_fitness"] is None:
                progress["best_fitness"] = 60.0  # Starting fitness
                progress["average_fitness"] = 45.0
            else:
                # Simulate convergence
                improvement = max(0, (config.target_spl - progress["best_fitness"]) * 0.05)
                progress["best_fitness"] += improvement
                progress["average_fitness"] += improvement * 0.7
            
            # Calculate convergence rate
            if generation > 0:
                progress["convergence_rate"] = improvement / max(0.1, progress["best_fitness"])
            
            # Estimate remaining time
            elapsed_time = (generation + 1) * 0.1
            remaining_generations = config.max_generations - generation - 1
            progress["estimated_time_remaining"] = remaining_generations * 0.1
            
            # Create generation update
            generation_data = OptimizationGeneration(
                generation=generation + 1,
                best_fitness=progress["best_fitness"],
                avg_fitness=progress["average_fitness"],
                best_configuration=sources,  # Simplified - in real implementation, this would be optimized
                convergence_rate=progress["convergence_rate"],
                estimated_time_remaining=progress["estimated_time_remaining"]
            )
            
            # Send update via WebSocket (if connected)
            try:
                await websocket_manager.broadcast_to_all({
                    "type": "optimization:generation",
                    "data": generation_data.dict()
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to send WebSocket update: {e}")
            
            # Check convergence
            if abs(progress["best_fitness"] - config.target_spl) <= config.tolerance:
                logger.info(f"🎯 Optimization {optimization_id} converged at generation {generation + 1}")
                break
        
        # Complete optimization
        if session["status"] != "cancelled":
            session["status"] = "completed"
            session["results"] = {
                "final_configuration": [source.dict() for source in sources],
                "convergence_data": [progress["best_fitness"]] * (generation + 1),
                "total_generations": generation + 1,
                "final_fitness": progress["best_fitness"],
                "target_achieved": abs(progress["best_fitness"] - config.target_spl) <= config.tolerance
            }
            
            # Send completion event
            try:
                await websocket_manager.broadcast_to_all({
                    "type": "optimization:complete",
                    "data": session["results"]
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to send completion event: {e}")
            
            logger.info(f"✅ Optimization {optimization_id} completed")
        
    except Exception as e:
        logger.error(f"❌ Optimization {optimization_id} failed: {e}")
        session["status"] = "failed"
        session["error"] = str(e)

def _estimate_optimization_time(config: OptimizationConfig) -> str:
    """Estimate optimization completion time"""
    # Rough estimation based on configuration
    base_time = config.population_size * config.max_generations * 0.01  # seconds
    
    if base_time < 60:
        return f"{int(base_time)} seconds"
    elif base_time < 3600:
        return f"{int(base_time / 60)} minutes"
    else:
        return f"{int(base_time / 3600)} hours"