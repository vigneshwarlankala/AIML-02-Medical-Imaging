import json
import unittest
from urllib.request import urlopen

BASE_URL = "http://127.0.0.1:8000"

class APISmokeTests(unittest.TestCase):

    def get_json(self, endpoint):
        with urlopen(BASE_URL + endpoint, timeout=15) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def test_api_health(self):
        data = self.get_json("/health")
        self.assertEqual(data.get("status"), "ready")
        self.assertTrue(data.get("model_loaded"))

    def test_model_info(self):
        data = self.get_json("/model-info")
        self.assertIsInstance(data, dict)
        self.assertTrue(data)

    def test_api_root(self):
        with urlopen(BASE_URL + "/", timeout=15) as response:
            self.assertEqual(response.status, 200)

if __name__ == "__main__":
    unittest.main()
