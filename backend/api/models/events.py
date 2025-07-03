"""
Event Models per WebSocket Communication
Type-safe event definitions per client-server communication
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import time

class EventType(str, Enum):
    """Event types for type-safe routing"""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    
    # Simulation events
    SIMULATION_START = "simulation:start"
    SIMULATION_STOP = "simulation:stop"
    SIMULATION_PARAMETER_UPDATE = "simulation:parameter_update"
    SIMULATION_PROGRESS = "simulation:progress"
    SIMULATION_SPL_CHUNK = "simulation:spl_chunk"
    SIMULATION_COMPLETE = "simulation:complete"
    
    # Project events
    PROJECT_LOAD = "project:load"
    PROJECT_SAVE = "project:save"
    PROJECT_JOIN = "project:join"
    PROJECT_LEAVE = "project:leave"
    
    # Optimization events
    OPTIMIZATION_START = "optimization:start"
    OPTIMIZATION_STOP = "optimization:stop"
    OPTIMIZATION_GENERATION = "optimization:generation"
    OPTIMIZATION_COMPLETE = "optimization:complete"
    
    # Interactive events
    SOURCE_MOVE = "source:move"
    SOURCE_ROTATE = "source:rotate"
    ROOM_VERTEX_MOVE = "room:vertex_move"
    
    # UI events
    UI_SOURCE_MOVED = "ui:source_moved"
    UI_ROOM_VERTEX_MOVED = "ui:room_vertex_moved"
    UI_SPL_PREVIEW = "ui:spl_preview"
    UI_PARAMETER_VALIDATION = "ui:parameter_validation"
    
    # Parameter feedback
    PARAMETER_VALIDATION = "parameter:validation"

class BaseEvent(BaseModel):
    """Base event model with common fields"""
    type: EventType
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = Field(default_factory=dict)

class ClientEvent(BaseEvent):
    """Events sent from client to server"""
    session_id: Optional[str] = None
    request_id: Optional[str] = None  # For tracking async responses
    
    @classmethod
    def parse_obj(cls, obj: Dict[str, Any]) -> 'ClientEvent':
        """Parse object with type validation"""
        if 'type' not in obj:
            raise ValueError("Event must have 'type' field")
        
        event_type = obj['type']
        if event_type not in EventType.__members__.values():
            raise ValueError(f"Unknown event type: {event_type}")
        
        return cls(
            type=EventType(event_type),
            data=obj.get('data', {}),
            session_id=obj.get('session_id'),
            request_id=obj.get('request_id'),
            timestamp=obj.get('timestamp', time.time())
        )

class ServerEvent(BaseEvent):
    """Events sent from server to client"""
    session_id: Optional[str] = None
    request_id: Optional[str] = None  # For async response correlation
    
    @classmethod
    def create_response(cls, request: ClientEvent, event_type: EventType, data: Dict[str, Any]) -> 'ServerEvent':
        """Create response event for a client request"""
        return cls(
            type=event_type,
            data=data,
            session_id=request.session_id,
            request_id=request.request_id,
            timestamp=time.time()
        )

# Specific event data models
class SplPreviewData(BaseModel):
    """SPL preview data for UI hover"""
    point: List[float]  # [x, y] coordinates
    spl_value: float
    frequency: float
    timestamp: float = Field(default_factory=time.time)

class SimulationParams(BaseModel):
    """Parameters for SPL simulation"""
    frequency: float = Field(ge=20, le=20000, description="Frequency in Hz")
    speed_of_sound: float = Field(ge=300, le=400, description="Speed of sound in m/s")
    grid_resolution: float = Field(ge=0.01, le=1.0, description="Grid resolution in meters")
    room_vertices: List[List[float]] = Field(description="Room boundary vertices")
    target_areas: List[Dict[str, Any]] = Field(default_factory=list)
    avoidance_areas: List[Dict[str, Any]] = Field(default_factory=list)

class SourceData(BaseModel):
    """Subwoofer source data"""
    id: str
    x: float = Field(description="X position in meters")
    y: float = Field(description="Y position in meters")
    spl_rms: float = Field(ge=50, le=150, description="SPL RMS in dB")
    gain_db: float = Field(ge=-60, le=20, description="Gain in dB")
    delay_ms: float = Field(ge=0, le=1000, description="Delay in milliseconds")
    angle: float = Field(ge=0, le=360, description="Rotation angle in degrees")
    polarity: int = Field(ge=-1, le=1, description="Polarity: -1 or 1")

class SplChunkData(BaseModel):
    """SPL map chunk for progressive loading"""
    X: List[List[float]]
    Y: List[List[float]]
    SPL: List[List[float]]
    chunk_index: int
    total_chunks: int
    is_last: bool = False

class OptimizationConfig(BaseModel):
    """Configuration for genetic algorithm optimization"""
    population_size: int = Field(ge=10, le=200, default=50)
    max_generations: int = Field(ge=10, le=1000, default=100)
    mutation_rate: float = Field(ge=0.001, le=0.1, default=0.01)
    crossover_rate: float = Field(ge=0.1, le=1.0, default=0.8)
    target_spl: float = Field(ge=60, le=120, default=85)
    tolerance: float = Field(ge=1, le=10, default=3)

class OptimizationGeneration(BaseModel):
    """Data for optimization generation update"""
    generation: int
    best_fitness: float
    avg_fitness: float
    best_configuration: List[SourceData]
    convergence_rate: float
    estimated_time_remaining: Optional[float] = None

class ErrorData(BaseModel):
    """Error event data"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None

class ConnectionData(BaseModel):
    """Connection event data"""
    session_id: str
    server_version: str
    timestamp: float
    capabilities: List[str] = Field(default_factory=lambda: [
        "real_time_spl",
        "optimization_streaming", 
        "multi_user_projects",
        "parameter_validation"
    ])

class ParameterValidation(BaseModel):
    """Parameter validation response"""
    parameter: str
    valid: bool
    message: Optional[str] = None
    suggested_value: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None

# Event factory functions
def create_error_event(session_id: str, code: str, message: str, details: Optional[Dict] = None) -> ServerEvent:
    """Create standardized error event"""
    return ServerEvent(
        type=EventType.ERROR,
        session_id=session_id,
        data=ErrorData(
            code=code,
            message=message,
            details=details or {}
        ).dict()
    )

def create_connection_event(session_id: str, server_version: str = "2.0.0") -> ServerEvent:
    """Create connection confirmation event"""
    return ServerEvent(
        type=EventType.CONNECTED,
        session_id=session_id,
        data=ConnectionData(
            session_id=session_id,
            server_version=server_version,
            timestamp=time.time()
        ).dict()
    )

def create_simulation_progress_event(session_id: str, progress: float, current_step: str, estimated_time: Optional[float] = None) -> ServerEvent:
    """Create simulation progress event"""
    return ServerEvent(
        type=EventType.SIMULATION_PROGRESS,
        session_id=session_id,
        data={
            "progress": progress,
            "current_step": current_step,
            "estimated_time": estimated_time,
            "timestamp": time.time()
        }
    )

def create_spl_chunk_event(session_id: str, chunk_data: SplChunkData) -> ServerEvent:
    """Create SPL chunk event for progressive loading"""
    return ServerEvent(
        type=EventType.SIMULATION_SPL_CHUNK,
        session_id=session_id,
        data=chunk_data.dict()
    )

def create_optimization_generation_event(session_id: str, generation_data: OptimizationGeneration) -> ServerEvent:
    """Create optimization generation update event"""
    return ServerEvent(
        type=EventType.OPTIMIZATION_GENERATION,
        session_id=session_id,
        data=generation_data.dict()
    )

def create_spl_preview_event(session_id: str, preview_data: SplPreviewData) -> ServerEvent:
    """Create SPL preview event for UI hover"""
    return ServerEvent(
        type=EventType.UI_SPL_PREVIEW,
        session_id=session_id,
        data=preview_data.dict()
    )

def create_parameter_validation_event(session_id: str, validation: ParameterValidation) -> ServerEvent:
    """Create parameter validation response event"""
    return ServerEvent(
        type=EventType.PARAMETER_VALIDATION,
        session_id=session_id,
        data=validation.dict()
    )