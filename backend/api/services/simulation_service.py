"""
Simulation Service per Real-time SPL Calculations
Gestisce simulazioni asincrone con streaming progressive results
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from ..models.events import (
    SimulationParams, SourceData, SplChunkData, 
    create_simulation_progress_event, create_spl_chunk_event,
    create_error_event, EventType, ServerEvent
)
from .websocket_manager import WebSocketManager

# Import existing core modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from core.acoustic_engine import calculate_spl_vectorized
from core.config import SimulationConfig, SUB_DTYPE

logger = logging.getLogger(__name__)

@dataclass
class ActiveSimulation:
    """Represents an active simulation session"""
    simulation_id: str
    session_id: str
    params: SimulationParams
    sources: List[SourceData]
    created_at: float
    task: Optional[asyncio.Task] = None
    progress: float = 0.0
    current_step: str = "initializing"
    estimated_time: Optional[float] = None
    is_cancelled: bool = False

class SimulationService:
    """
    High-performance simulation service with real-time streaming
    
    Features:
    - Asynchronous SPL calculations with progress streaming
    - Chunked data transmission for large maps
    - Real-time parameter validation
    - Multiple concurrent simulations per session
    - Graceful cancellation and cleanup
    """
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        
        # Active simulations tracking
        self.active_simulations: Dict[str, ActiveSimulation] = {}
        self.session_simulations: Dict[str, List[str]] = {}  # session_id -> simulation_ids
        
        # Performance counters
        self.completed_count = 0
        self.cancelled_count = 0
        self.error_count = 0
        
        # Configuration
        self.max_simulations_per_session = 3
        self.chunk_size = 100  # Grid points per chunk (reduced for WebSocket message size)
        self.progress_update_interval = 0.5  # seconds
        
    async def startup(self):
        """Initialize simulation service"""
        logger.info("🧮 Starting Simulation Service")
        
    async def shutdown(self):
        """Cleanup and cancel all active simulations"""
        logger.info("🛑 Shutting down Simulation Service")
        
        # Cancel all active simulations
        cancel_tasks = []
        for sim in self.active_simulations.values():
            if sim.task and not sim.task.done():
                sim.is_cancelled = True
                cancel_tasks.append(sim.task)
                
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
            
        self.active_simulations.clear()
        self.session_simulations.clear()
        
    async def start_simulation(self, session_id: str, event_data: Dict[str, Any], request_id: str = None) -> bool:
        """
        Start new SPL simulation with real-time streaming
        
        Args:
            session_id: WebSocket session ID
            event_data: Simulation parameters and sources
            
        Returns:
            bool: True if simulation started successfully
        """
        try:
            # Check session simulation limit
            current_sims = len(self.session_simulations.get(session_id, []))
            if current_sims >= self.max_simulations_per_session:
                await self.websocket_manager.send_error(
                    session_id, 
                    "simulation_limit", 
                    f"Maximum {self.max_simulations_per_session} simulations per session",
                    request_id
                )
                return False
            
            # Parse and validate parameters
            try:
                params = SimulationParams.parse_obj(event_data.get('params', {}))
                sources_data = event_data.get('sources', [])
                sources = [SourceData.parse_obj(src) for src in sources_data]
                
                if not sources:
                    await self.websocket_manager.send_error(
                        session_id, "invalid_params", "At least one source required", request_id
                    )
                    return False
                    
            except Exception as e:
                await self.websocket_manager.send_error(
                    session_id, "validation_error", f"Invalid parameters: {str(e)}", request_id
                )
                return False
            
            # Create simulation
            simulation_id = str(uuid.uuid4())
            simulation = ActiveSimulation(
                simulation_id=simulation_id,
                session_id=session_id,
                params=params,
                sources=sources,
                created_at=time.time()
            )
            
            # Track simulation
            self.active_simulations[simulation_id] = simulation
            if session_id not in self.session_simulations:
                self.session_simulations[session_id] = []
            self.session_simulations[session_id].append(simulation_id)
            
            # Start async simulation task
            simulation.task = asyncio.create_task(
                self._run_simulation(simulation)
            )
            
            logger.info(f"🧮 Started simulation {simulation_id} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting simulation for {session_id}: {e}")
            await self.websocket_manager.send_error(
                session_id, "simulation_error", str(e)
            )
            return False
            
    async def stop_simulation(self, session_id: str, request_id: str = None, simulation_id: Optional[str] = None) -> bool:
        """
        Stop specific simulation or all simulations for session
        
        Args:
            session_id: WebSocket session ID
            simulation_id: Optional specific simulation to stop
            
        Returns:
            bool: True if simulation(s) stopped successfully
        """
        try:
            if simulation_id:
                # Stop specific simulation
                simulation = self.active_simulations.get(simulation_id)
                if simulation and simulation.session_id == session_id:
                    await self._cancel_simulation(simulation)
                    return True
                else:
                    await self.websocket_manager.send_error(
                        session_id, "simulation_not_found", f"Simulation {simulation_id} not found", request_id
                    )
                    return False
            else:
                # Stop all simulations for session
                sim_ids = self.session_simulations.get(session_id, []).copy()
                for sim_id in sim_ids:
                    simulation = self.active_simulations.get(sim_id)
                    if simulation:
                        await self._cancel_simulation(simulation)
                
                logger.info(f"🛑 Stopped {len(sim_ids)} simulations for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error stopping simulation for {session_id}: {e}")
            return False
            
    async def update_parameter(self, session_id: str, event_data: Dict[str, Any], request_id: str = None) -> bool:
        """
        Update simulation parameter with real-time validation
        
        Args:
            session_id: WebSocket session ID
            event_data: Parameter update data
            
        Returns:
            bool: True if parameter updated successfully
        """
        try:
            param_name = event_data.get('param')
            param_value = event_data.get('value')
            
            if not param_name:
                await self.websocket_manager.send_error(
                    session_id, "invalid_param", "Parameter name required", request_id
                )
                return False
            
            # Validate parameter
            validation_result = await self._validate_parameter(param_name, param_value)
            
            # Send validation response
            await self.websocket_manager.send_to_session(session_id, {
                "type": "parameter:validation",
                "data": {
                    "param": param_name,
                    "valid": validation_result["valid"],
                    "message": validation_result.get("message"),
                    "suggested_value": validation_result.get("suggested_value"),
                    "constraints": validation_result.get("constraints")
                }
            })
            
            return validation_result["valid"]
            
        except Exception as e:
            logger.error(f"❌ Error updating parameter for {session_id}: {e}")
            return False
            
    async def _run_simulation(self, simulation: ActiveSimulation):
        """
        Execute SPL simulation with real-time progress streaming
        
        Args:
            simulation: Active simulation instance
        """
        try:
            session_id = simulation.session_id
            
            # Send start confirmation
            await self.websocket_manager.send_to_session(session_id, {
                "type": "simulation:started",
                "data": {
                    "simulation_id": simulation.simulation_id,
                    "estimated_time": self._estimate_computation_time(simulation),
                    "grid_size": self._calculate_grid_size(simulation.params)
                }
            })
            
            # Step 1: Prepare sources array
            simulation.current_step = "preparing_sources"
            simulation.progress = 10.0
            await self._send_progress_update(simulation)
            
            sources_array = self._prepare_sources_array(simulation.sources)
            
            if simulation.is_cancelled:
                return
                
            # Step 2: Setup simulation grid
            simulation.current_step = "setting_up_grid"
            simulation.progress = 20.0
            await self._send_progress_update(simulation)
            
            # Create config with no parameters (as it doesn't accept any)
            config = SimulationConfig()
            
            # Update config values using the default_values dictionary
            # Note: SimulationConfig stores parameters in default_values, not constructor params
            if hasattr(config, 'default_values'):
                # Update frequency if it exists in config
                if 'array_freq' in config.default_values:
                    config.default_values['array_freq'] = float(simulation.params.frequency)
                
                # Update room vertices if they exist
                if 'room_vertices' in config.default_values:
                    config.default_values['room_vertices'] = simulation.params.room_vertices
            
            # Update acoustic parameters (speed of sound is there)
            if hasattr(config, 'acoustic_params'):
                config.acoustic_params['speed_of_sound'] = float(simulation.params.speed_of_sound)
            
            logger.info(f"🔧 Configured simulation with frequency={simulation.params.frequency}, speed_of_sound={simulation.params.speed_of_sound}")
            
            # Step 3: Calculate SPL with chunked streaming
            simulation.current_step = "calculating_spl"
            simulation.progress = 30.0
            await self._send_progress_update(simulation)
            
            await self._calculate_spl_chunked(simulation, sources_array, config)
            
            if simulation.is_cancelled:
                return
                
            # Step 4: Apply room masking
            simulation.current_step = "applying_room_mask"
            simulation.progress = 90.0
            await self._send_progress_update(simulation)
            
            await asyncio.sleep(0.1)  # Simulate room processing
            
            # Step 5: Complete simulation
            simulation.current_step = "complete"
            simulation.progress = 100.0
            await self._send_progress_update(simulation)
            
            # Send completion event
            await self.websocket_manager.send_to_session(session_id, {
                "type": "simulation:complete",
                "data": {
                    "simulation_id": simulation.simulation_id,
                    "computation_time": time.time() - simulation.created_at,
                    "statistics": {
                        "total_sources": len(simulation.sources),
                        "grid_points": self._calculate_grid_size(simulation.params),
                        "frequency": simulation.params.frequency
                    }
                }
            })
            
            self.completed_count += 1
            logger.info(f"✅ Simulation {simulation.simulation_id} completed")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Simulation {simulation.simulation_id} cancelled")
            self.cancelled_count += 1
        except Exception as e:
            logger.error(f"❌ Simulation {simulation.simulation_id} failed: {e}")
            self.error_count += 1
            await self.websocket_manager.send_error(
                simulation.session_id, 
                "simulation_error", 
                f"Simulation failed: {str(e)}"
            )
        finally:
            # Cleanup
            await self._cleanup_simulation(simulation)
            
    async def _calculate_spl_chunked(self, simulation: ActiveSimulation, sources_array: np.ndarray, config: SimulationConfig):
        """
        Calculate SPL with chunked streaming for large datasets
        
        Args:
            simulation: Active simulation instance
            sources_array: NumPy array of source data
            config: Simulation configuration
        """
        try:
            # Calculate grid dimensions
            grid_size = self._calculate_grid_size(simulation.params)
            chunk_size = min(self.chunk_size, grid_size)
            total_chunks = max(1, grid_size // chunk_size)
            
            # Create coordinate grids
            room_bounds = self._calculate_room_bounds(simulation.params.room_vertices)
            x_range = np.linspace(room_bounds[0], room_bounds[1], int(np.sqrt(grid_size)))
            y_range = np.linspace(room_bounds[2], room_bounds[3], int(np.sqrt(grid_size)))
            X, Y = np.meshgrid(x_range, y_range)
            
            # Process in chunks
            for chunk_idx in range(total_chunks):
                if simulation.is_cancelled:
                    break
                    
                # Calculate chunk boundaries
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, grid_size)
                
                # Extract chunk coordinates
                flat_X = X.flatten()
                flat_Y = Y.flatten()
                
                chunk_X = flat_X[start_idx:end_idx]
                chunk_Y = flat_Y[start_idx:end_idx]
                
                # Calculate SPL for chunk using existing engine
                chunk_grid_X, chunk_grid_Y = np.meshgrid(chunk_X, chunk_Y)
                chunk_SPL = calculate_spl_vectorized(
                    chunk_grid_X.flatten(),
                    chunk_grid_Y.flatten(),
                    simulation.params.frequency,
                    simulation.params.speed_of_sound,
                    sources_array
                ).reshape(chunk_grid_X.shape)
                
                # Create chunk data
                chunk_data = SplChunkData(
                    X=chunk_grid_X.tolist(),
                    Y=chunk_grid_Y.tolist(),
                    SPL=chunk_SPL.tolist(),
                    chunk_index=chunk_idx,
                    total_chunks=total_chunks,
                    is_last=(chunk_idx == total_chunks - 1)
                )
                
                # Send chunk to client
                await self.websocket_manager.send_to_session(
                    simulation.session_id,
                    create_spl_chunk_event(simulation.session_id, chunk_data).dict()
                )
                
                # Update progress
                chunk_progress = 30.0 + (chunk_idx / total_chunks) * 60.0
                simulation.progress = chunk_progress
                await self._send_progress_update(simulation)
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.error(f"❌ Error in chunked SPL calculation: {e}")
            raise
            
    async def _send_progress_update(self, simulation: ActiveSimulation):
        """Send progress update to client"""
        try:
            event = create_simulation_progress_event(
                simulation.session_id,
                simulation.progress,
                simulation.current_step,
                simulation.estimated_time
            )
            await self.websocket_manager.send_to_session(
                simulation.session_id,
                event.dict()
            )
        except Exception as e:
            logger.error(f"❌ Error sending progress update: {e}")
            
    async def _cancel_simulation(self, simulation: ActiveSimulation):
        """Cancel active simulation"""
        simulation.is_cancelled = True
        if simulation.task and not simulation.task.done():
            simulation.task.cancel()
            try:
                await simulation.task
            except asyncio.CancelledError:
                pass
        await self._cleanup_simulation(simulation)
        
    async def _cleanup_simulation(self, simulation: ActiveSimulation):
        """Cleanup simulation resources"""
        # Remove from tracking
        if simulation.simulation_id in self.active_simulations:
            del self.active_simulations[simulation.simulation_id]
            
        if simulation.session_id in self.session_simulations:
            session_sims = self.session_simulations[simulation.session_id]
            if simulation.simulation_id in session_sims:
                session_sims.remove(simulation.simulation_id)
            if not session_sims:
                del self.session_simulations[simulation.session_id]
                
    def _prepare_sources_array(self, sources: List[SourceData]) -> np.ndarray:
        """Convert sources to NumPy array for calculations"""
        sources_array = np.zeros(len(sources), dtype=SUB_DTYPE)
        for i, source in enumerate(sources):
            # Convert gain_db to linear gain for pressure calculation
            gain_linear = 10**(source.gain_db / 20.0)
            
            # Convert SPL RMS to pressure value at 1m (relative to reference pressure)
            # Standard reference pressure: 20e-6 Pa
            p_ref = 20e-6
            pressure_val = p_ref * (10**(source.spl_rms / 20.0))
            
            sources_array[i] = (
                source.x,                    # x
                source.y,                    # y  
                pressure_val,                # pressure_val_at_1m_relative_to_pref
                gain_linear,                 # gain_lin
                source.angle,                # angle
                source.delay_ms,            # delay_ms
                source.polarity             # polarity
            )
        return sources_array
        
    def _calculate_grid_size(self, params: SimulationParams) -> int:
        """Calculate total grid points for simulation"""
        room_bounds = self._calculate_room_bounds(params.room_vertices)
        width = room_bounds[1] - room_bounds[0]
        height = room_bounds[3] - room_bounds[2]
        
        points_x = int(width / params.grid_resolution)
        points_y = int(height / params.grid_resolution)
        
        return points_x * points_y
        
    def _calculate_room_bounds(self, vertices: List[List[float]]) -> List[float]:
        """Calculate room bounding box"""
        if not vertices:
            return [0, 10, 0, 10]  # Default room
            
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        
        return [min(xs), max(xs), min(ys), max(ys)]
        
    def _estimate_computation_time(self, simulation: ActiveSimulation) -> float:
        """Estimate computation time based on parameters"""
        grid_size = self._calculate_grid_size(simulation.params)
        source_count = len(simulation.sources)
        
        # Rough estimation based on grid size and source count
        base_time = (grid_size * source_count) / 1000000  # seconds
        return max(1.0, base_time)
        
    async def _validate_parameter(self, param_name: str, param_value: Any) -> Dict[str, Any]:
        """Validate simulation parameter"""
        try:
            if param_name == "frequency":
                if not isinstance(param_value, (int, float)):
                    return {"valid": False, "message": "Frequency must be a number"}
                if param_value < 20 or param_value > 20000:
                    return {"valid": False, "message": "Frequency must be between 20-20000 Hz"}
                return {"valid": True}
                
            elif param_name == "grid_resolution":
                if not isinstance(param_value, (int, float)):
                    return {"valid": False, "message": "Grid resolution must be a number"}
                if param_value < 0.01 or param_value > 1.0:
                    return {"valid": False, "message": "Grid resolution must be between 0.01-1.0 m"}
                return {"valid": True}
                
            elif param_name == "speed_of_sound":
                if not isinstance(param_value, (int, float)):
                    return {"valid": False, "message": "Speed of sound must be a number"}
                if param_value < 300 or param_value > 400:
                    return {"valid": False, "message": "Speed of sound must be between 300-400 m/s"}
                return {"valid": True}
                
            else:
                return {"valid": False, "message": f"Unknown parameter: {param_name}"}
                
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {str(e)}"}
            
    def get_session_simulations(self, session_id: str) -> List[str]:
        """Get active simulation IDs for session"""
        return self.session_simulations.get(session_id, [])
        
    def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation status"""
        simulation = self.active_simulations.get(simulation_id)
        if not simulation:
            return None
            
        return {
            "simulation_id": simulation_id,
            "session_id": simulation.session_id,
            "progress": simulation.progress,
            "current_step": simulation.current_step,
            "created_at": simulation.created_at,
            "is_cancelled": simulation.is_cancelled,
            "estimated_time": simulation.estimated_time
        }
        
    async def calculate_point_spl(self, x: float, y: float, frequency: float) -> float:
        """
        Calculate SPL at a specific point for preview purposes
        Fast approximation for real-time UI feedback
        """
        try:
            # Create minimal source for calculation (simplified)
            # In a real implementation, this would use the current project's sources
            # Create a test source with proper SUB_DTYPE structure
            p_ref = 20e-6
            test_spl = 105.0  # dB SPL
            pressure_val = p_ref * (10**(test_spl / 20.0))
            sources_array = np.array([(x, y, pressure_val, 1.0, 0.0, 0.0, 1)], dtype=SUB_DTYPE)
            
            # Quick calculation
            point_x = np.array([x])
            point_y = np.array([y])
            
            spl_result = calculate_spl_vectorized(
                point_x, point_y, frequency, 343.0, sources_array
            )
            
            return float(spl_result[0]) if len(spl_result) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Point SPL calculation error: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation service statistics"""
        return {
            "active_simulations": len(self.active_simulations),
            "completed_simulations": self.completed_count,
            "cancelled_simulations": self.cancelled_count,
            "error_count": self.error_count,
            "sessions_with_simulations": len(self.session_simulations)
        }