import unittest
import asyncio
from backend.api.services.websocket_manager import WebSocketManager
from backend.api.services.simulation_service import SimulationService


class TestSimulationService(unittest.TestCase):
    def test_calculate_point_spl(self):
        manager = WebSocketManager()
        service = SimulationService(manager)
        result = asyncio.run(service.calculate_point_spl(0.0, 0.0, 60.0))
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
