import unittest
from backend.api.models.events import create_connection_event, EventType


class TestEvents(unittest.TestCase):
    def test_create_connection_event(self):
        event = create_connection_event('session1', '1.0.0')
        self.assertEqual(event.type, EventType.CONNECTED)
        self.assertEqual(event.session_id, 'session1')
        self.assertIn('server_version', event.data)


if __name__ == '__main__':
    unittest.main()
