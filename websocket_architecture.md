# ⚡ Architettura WebSocket Real-time

## 🎯 Obiettivo
Creare comunicazione **bidirezionale real-time** tra frontend React e backend Python per:
- **SPL calculations streaming** (aggiornamenti progressivi durante calcolo)
- **Optimization progress** (live updates algoritmi genetici)
- **Interactive parameter changes** (instant feedback su modifiche UI)
- **Multi-user collaboration** (future: shared sessions)

## 🏗️ Architettura Overview

```
┌─────────────────┐    WebSocket     ┌──────────────────┐
│   React Client  │ ←─────────────→  │  FastAPI Server  │
│                 │    (Socket.IO)   │                  │
│  - UI Updates   │                  │  - SPL Engine    │
│  - User Input   │                  │  - Optimization  │
│  - Real-time    │                  │  - State Sync    │
│    Feedback     │                  │  - Broadcasting  │
└─────────────────┘                  └──────────────────┘
         │                                      │
         │                                      │
    ┌────▼────┐                          ┌─────▼─────┐
    │ Zustand │                          │   Core    │
    │  Store  │                          │ Simulation│
    │         │                          │  Modules  │
    └─────────┘                          └───────────┘
```

## 📡 Protocolli di Comunicazione

### Event Types (Client → Server)
```typescript
interface ClientEvents {
  // Project Management
  'project:load': { projectId: string }
  'project:save': { projectData: Project }
  
  // Simulation Control
  'simulation:start': { params: SimulationParams }
  'simulation:stop': {}
  'simulation:parameter_update': { param: string, value: any }
  
  // Real-time Interaction
  'source:move': { sourceId: string, x: number, y: number }
  'source:rotate': { sourceId: string, angle: number }
  'room:vertex_move': { vertexId: string, x: number, y: number }
  
  // Optimization
  'optimization:start': { config: OptimizationConfig }
  'optimization:stop': {}
}
```

### Event Types (Server → Client)
```typescript
interface ServerEvents {
  // Connection
  'connected': { sessionId: string }
  'error': { message: string, code: string }
  
  // Simulation Updates
  'simulation:progress': { 
    progress: number,           // 0-100%
    currentStep: string,       // "calculating_spl" | "applying_room_mask"
    estimatedTime: number      // seconds remaining
  }
  
  'simulation:spl_chunk': {
    X: number[][],            // Grid chunk
    Y: number[][],
    SPL: number[][],
    chunkIndex: number,       // For progressive loading
    totalChunks: number
  }
  
  'simulation:complete': {
    splMap: SplMapData,
    statistics: SimulationStats,
    computationTime: number
  }
  
  // Optimization Updates
  'optimization:generation': {
    generation: number,
    bestFitness: number,
    avgFitness: number,
    bestConfiguration: SourceConfiguration[]
  }
  
  'optimization:complete': {
    finalConfiguration: SourceConfiguration[],
    convergenceData: number[],
    totalGenerations: number
  }
  
  // Real-time Parameter Feedback
  'parameter:validation': {
    param: string,
    valid: boolean,
    message?: string,
    suggestedValue?: any
  }
  
  // Live UI Updates
  'ui:source_moved': { sourceId: string, x: number, y: number }
  'ui:spl_preview': { 
    point: [number, number], 
    splValue: number 
  }
}
```

## ⚡ Performance Optimizations

### 1. Chunked SPL Streaming
```python
# Backend: Stream SPL calculation in chunks
async def stream_spl_calculation(websocket, params):
    total_chunks = calculate_chunks_needed(params.grid_resolution)
    
    for chunk_idx in range(total_chunks):
        # Calculate SPL for current chunk
        X_chunk, Y_chunk, SPL_chunk = calculate_spl_chunk(chunk_idx, params)
        
        await websocket.send_json({
            "type": "simulation:spl_chunk",
            "data": {
                "X": X_chunk.tolist(),
                "Y": Y_chunk.tolist(), 
                "SPL": SPL_chunk.tolist(),
                "chunkIndex": chunk_idx,
                "totalChunks": total_chunks
            }
        })
        
        # Allow other tasks to run
        await asyncio.sleep(0.01)
```

### 2. Debounced Parameter Updates
```typescript
// Frontend: Debounce rapid parameter changes
const useParameterUpdate = () => {
  const debouncedUpdate = useMemo(
    () => debounce((param: string, value: any) => {
      socket.emit('simulation:parameter_update', { param, value });
    }, 300),
    [socket]
  );
  
  return debouncedUpdate;
};
```

### 3. Binary Data for Large Arrays
```python
# For very large SPL maps, use binary data
import msgpack

async def send_large_spl_data(websocket, spl_data):
    # Compress with msgpack for efficiency
    compressed = msgpack.packb({
        "X": spl_data.X,
        "Y": spl_data.Y, 
        "SPL": spl_data.SPL
    })
    
    await websocket.send_bytes(compressed)
```

## 🔄 State Synchronization Strategy

### Client-side State Management
```typescript
// Zustand store with WebSocket integration
interface SimulationStore {
  // Connection state
  isConnected: boolean;
  connectionError: string | null;
  
  // Simulation state
  currentSimulation: SimulationState | null;
  splMap: SplMapData | null;
  isSimulating: boolean;
  progress: number;
  
  // Real-time updates
  pendingUpdates: Map<string, any>;
  lastUpdateTime: number;
  
  // Actions
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
  updateParameter: (param: string, value: any) => void;
  applySplChunk: (chunk: SplChunk) => void;
}
```

### Conflict Resolution
```typescript
// Handle concurrent updates
const handleParameterConflict = (
  localValue: any, 
  serverValue: any, 
  timestamp: number
) => {
  const timeDiff = Date.now() - timestamp;
  
  // If server update is very recent, prefer server
  if (timeDiff < 1000) {
    return serverValue;
  }
  
  // Otherwise, keep local value and sync to server
  return localValue;
};
```

## 🚀 Implementation Benefits

### Real-time Responsiveness
- **~50ms latency** per parameter update
- **Progressive SPL loading** invece di wait completo
- **Live optimization visualization** con generazione-per-generazione
- **Instant validation feedback** per input utente

### User Experience
- **Smooth interactions** durante drag & drop sources
- **Real-time SPL preview** on hover
- **Progress indicators** per long-running operations
- **Offline resilience** con automatic reconnection

### Scalability
- **Multiple concurrent simulations** per user
- **Room for multi-user features** future
- **Efficient bandwidth usage** con compression
- **Graceful degradation** se WebSocket fails

Questa architettura garantisce **reattività ottimale** mantenendo performance elevate! 🎯