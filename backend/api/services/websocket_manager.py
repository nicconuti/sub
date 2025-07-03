"""
WebSocket Manager ottimizzato per performance e scalabilità
Gestisce connessioni multiple, broadcasting, e session management
"""

import asyncio
import json
import logging
import time
import uuid
import msgpack
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

@dataclass
class WebSocketSession:
    """Represents an active WebSocket session"""
    session_id: str
    websocket: WebSocket
    connected_at: float
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    last_ping: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

class WebSocketManager:
    """
    High-performance WebSocket manager per real-time communication
    
    Features:
    - Connection pooling con session management
    - Broadcasting ottimizzato per gruppi/progetti
    - Health monitoring con ping/pong
    - Error handling e graceful disconnection
    - Message queueing per offline clients
    """
    
    def __init__(self):
        # Core connection tracking
        self.active_connections: Dict[str, WebSocketSession] = {}
        self.project_sessions: Dict[str, Set[str]] = {}  # project_id -> session_ids
        
        # Performance monitoring
        self.session_count = 0
        self.message_count = 0
        self.error_count = 0
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10   # seconds
        self.max_connections = 1000
        
    async def startup(self):
        """Initialize background tasks"""
        logger.info("🚀 Starting WebSocket manager")
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def shutdown(self):
        """Cleanup and close all connections"""
        logger.info("🛑 Shutting down WebSocket manager")
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        # Close all connections gracefully
        close_tasks = []
        for session in self.active_connections.values():
            try:
                close_tasks.append(session.websocket.close())
            except:
                pass
                
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
        self.active_connections.clear()
        self.project_sessions.clear()
        
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept new WebSocket connection and create session
        
        Returns:
            session_id: Unique identifier for this session
        """
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            raise Exception("Connection limit exceeded")
            
        await websocket.accept()
        
        # Create session
        session_id = str(uuid.uuid4())
        session = WebSocketSession(
            session_id=session_id,
            websocket=websocket,
            connected_at=time.time()
        )
        
        self.active_connections[session_id] = session
        self.session_count += 1
        
        logger.info(f"✅ WebSocket connected: {session_id} (total: {len(self.active_connections)})")
        return session_id
        
    async def disconnect(self, session_id: str):
        """
        Disconnect session and cleanup
        """
        session = self.active_connections.get(session_id)
        if not session:
            return
            
        # Remove from project group
        if session.project_id:
            self._remove_from_project(session_id, session.project_id)
            
        # Remove from active connections
        del self.active_connections[session_id]
        
        logger.info(f"🔌 WebSocket disconnected: {session_id} (total: {len(self.active_connections)})")
        
    def _remove_from_project(self, session_id: str, project_id: str):
        """Remove session from project group"""
        if project_id in self.project_sessions:
            self.project_sessions[project_id].discard(session_id)
            if not self.project_sessions[project_id]:
                del self.project_sessions[project_id]
                
    async def join_project(self, session_id: str, project_id: str):
        """Add session to project group for broadcasting"""
        session = self.active_connections.get(session_id)
        if not session:
            return
            
        # Remove from previous project
        if session.project_id:
            self._remove_from_project(session_id, session.project_id)
            
        # Add to new project
        session.project_id = project_id
        if project_id not in self.project_sessions:
            self.project_sessions[project_id] = set()
        self.project_sessions[project_id].add(session_id)
        
        logger.info(f"👥 Session {session_id} joined project {project_id}")
        
    async def send_to_session(self, session_id: str, message: Dict) -> bool:
        """
        Send message to specific session
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        session = self.active_connections.get(session_id)
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return False
            
        try:
            await session.websocket.send_text(json.dumps(message))
            self.message_count += 1
            return True
        except WebSocketDisconnect:
            await self.disconnect(session_id)
            return False
        except Exception as e:
            logger.error(f"❌ Error sending to {session_id}: {e}")
            self.error_count += 1
            await self.disconnect(session_id)
            return False
            
    async def send_error(self, session_id: str, error_code: str, message: str):
        """Send error message to session"""
        await self.send_to_session(session_id, {
            "type": "error",
            "data": {
                "code": error_code,
                "message": message,
                "timestamp": time.time()
            }
        })
        
    async def broadcast_to_project(self, project_id: str, message: Dict, exclude_session: Optional[str] = None):
        """
        Broadcast message to all sessions in a project
        
        Args:
            project_id: Target project ID
            message: Message to broadcast
            exclude_session: Optional session to exclude from broadcast
        """
        if project_id not in self.project_sessions:
            return
            
        session_ids = self.project_sessions[project_id].copy()
        if exclude_session:
            session_ids.discard(exclude_session)
            
        # Send to all sessions concurrently
        tasks = []
        for session_id in session_ids:
            tasks.append(self.send_to_session(session_id, message))
            
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.debug(f"📡 Broadcast to project {project_id}: {success_count}/{len(tasks)} delivered")
            
    async def broadcast_to_all(self, message: Dict, exclude_session: Optional[str] = None):
        """Broadcast message to all connected sessions"""
        session_ids = list(self.active_connections.keys())
        if exclude_session:
            session_ids = [sid for sid in session_ids if sid != exclude_session]
            
        tasks = [self.send_to_session(sid, message) for sid in session_ids]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.debug(f"📡 Global broadcast: {success_count}/{len(tasks)} delivered")
            
    async def send_chunked_data(self, session_id: str, event_type: str, data: Dict, chunk_size: int = 1000):
        """
        Send large data in chunks per performance
        
        Useful per SPL maps, optimization results, etc.
        """
        session = self.active_connections.get(session_id)
        if not session:
            return False
            
        try:
            # Send start message
            await self.send_to_session(session_id, {
                "type": f"{event_type}:start",
                "data": {
                    "totalSize": len(str(data)),
                    "chunkSize": chunk_size
                }
            })
            
            # Send data in chunks
            data_str = json.dumps(data)
            for i in range(0, len(data_str), chunk_size):
                chunk = data_str[i:i + chunk_size]
                await self.send_to_session(session_id, {
                    "type": f"{event_type}:chunk",
                    "data": {
                        "chunk": chunk,
                        "index": i // chunk_size,
                        "isLast": i + chunk_size >= len(data_str)
                    }
                })
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)
                
            # Send completion message
            await self.send_to_session(session_id, {
                "type": f"{event_type}:complete",
                "data": {}
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending chunked data to {session_id}: {e}")
            return False
            
    async def _health_check_loop(self):
        """Background task per health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                # Check all connections
                current_time = time.time()
                stale_sessions = []
                
                for session_id, session in self.active_connections.items():
                    # Check if session is stale
                    if current_time - session.last_ping > self.ping_timeout * 2:
                        stale_sessions.append(session_id)
                        continue
                        
                    # Send ping
                    try:
                        await session.websocket.ping()
                        session.last_ping = current_time
                    except:
                        stale_sessions.append(session_id)
                        
                # Cleanup stale sessions
                for session_id in stale_sessions:
                    logger.warning(f"🩺 Removing stale session: {session_id}")
                    await self.disconnect(session_id)
                    
                if len(self.active_connections) > 0:
                    logger.debug(f"🩺 Health check: {len(self.active_connections)} active connections")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                
    async def _cleanup_loop(self):
        """Background task per periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Log statistics
                logger.info(f"📊 WebSocket stats: "
                          f"connections={len(self.active_connections)}, "
                          f"projects={len(self.project_sessions)}, "
                          f"messages={self.message_count}, "
                          f"errors={self.error_count}")
                
                # Reset counters periodically
                if self.message_count > 100000:
                    self.message_count = 0
                    self.error_count = 0
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                
    def get_stats(self) -> Dict:
        """Get current WebSocket statistics"""
        return {
            "active_connections": len(self.active_connections),
            "project_groups": len(self.project_sessions),
            "total_sessions": self.session_count,
            "total_messages": self.message_count,
            "total_errors": self.error_count,
            "uptime": time.time()
        }
        
    async def send_binary_data(self, session_id: str, data: Dict, compression: str = "msgpack") -> bool:
        """
        Send large data using binary compression for performance
        Useful for very large SPL maps and optimization results
        """
        session = self.active_connections.get(session_id)
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return False
            
        try:
            if compression == "msgpack":
                # Compress with msgpack for efficiency
                compressed_data = msgpack.packb(data)
                await session.websocket.send_bytes(compressed_data)
            else:
                # Fallback to JSON
                await session.websocket.send_text(json.dumps(data))
                
            self.message_count += 1
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(session_id)
            return False
        except Exception as e:
            logger.error(f"❌ Error sending binary data to {session_id}: {e}")
            self.error_count += 1
            await self.disconnect(session_id)
            return False
            
    async def send_large_spl_data(self, session_id: str, spl_data: Dict) -> bool:
        """
        Send large SPL map using optimal method based on data size
        Automatically chooses between chunked JSON and binary compression
        """
        try:
            # Estimate data size
            data_size = len(str(spl_data))
            
            # For very large data (>1MB), use binary compression
            if data_size > 1_000_000:
                logger.info(f"📊 Sending large SPL data ({data_size} bytes) using msgpack compression")
                return await self.send_binary_data(session_id, spl_data, "msgpack")
            
            # For medium data (>100KB), use chunked transmission
            elif data_size > 100_000:
                logger.info(f"📊 Sending medium SPL data ({data_size} bytes) using chunks")
                return await self.send_chunked_data(session_id, "spl_data", spl_data)
            
            # For small data, use regular JSON
            else:
                return await self.send_to_session(session_id, {
                    "type": "simulation:spl_complete",
                    "data": spl_data
                })
                
        except Exception as e:
            logger.error(f"❌ Error sending large SPL data to {session_id}: {e}")
            return False