import unittest

from observability import get_health_snapshot, format_health_text


class ObservabilityTests(unittest.TestCase):
    def test_health_snapshot_text(self):
        snapshot = get_health_snapshot()
        text = format_health_text(snapshot)
        self.assertIn("Status: OK", text)
        self.assertIn("Uptime:", text)


if __name__ == "__main__":
    unittest.main()
