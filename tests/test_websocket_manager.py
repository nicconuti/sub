import unittest
import asyncio
from backend.api.services.websocket_manager import WebSocketManager


class DummyWebSocket:
    def __init__(self):
        self.accepted = False
        self.closed = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code=1000, reason=""):
        self.closed = True

    async def ping(self):
        pass


class TestWebSocketManager(unittest.TestCase):
    def test_connect_and_disconnect(self):
        manager = WebSocketManager()
        ws = DummyWebSocket()
        session_id = asyncio.run(manager.connect(ws))
        self.assertTrue(ws.accepted)
        self.assertIn(session_id, manager.active_connections)
        asyncio.run(manager.disconnect(session_id))
        self.assertNotIn(session_id, manager.active_connections)

    def test_join_project_and_send(self):
        manager = WebSocketManager()
        ws = DummyWebSocket()
        session_id = asyncio.run(manager.connect(ws))
        asyncio.run(manager.join_project(session_id, "proj1"))
        self.assertIn("proj1", manager.project_sessions)
        asyncio.run(manager.send_to_session(session_id, {"hello": "world"}))
        self.assertTrue(ws.sent)
        asyncio.run(manager.disconnect(session_id))


if __name__ == '__main__':
    unittest.main()
