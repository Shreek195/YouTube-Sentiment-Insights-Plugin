import unittest
from flask_api.main import create_app  # import the Flask app instance directly

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app.test_client()  # directly use app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the YouTube Sentiment Insights Flask API', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', json={'comments': ['I love this!']})
        self.assertEqual(response.status_code, 200)

        json_data = response.get_json()
        self.assertIsInstance(json_data, list)
        self.assertGreater(len(json_data), 0)

        for item in json_data:
            self.assertIn(item['sentiment'], [-1, 0, 1], "Sentiment should be -1, 0, or 1")

if __name__ == '__main__':
    unittest.main()
