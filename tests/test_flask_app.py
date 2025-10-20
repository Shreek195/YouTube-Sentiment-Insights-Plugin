import unittest
from flask_api.main import create_app

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create app in testing mode → skips MLflow/model loading
        cls.client = create_app(testing=True).test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Match the actual <h1> text from main.py
        self.assertIn(b'<h1>Welcome</h1>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', json={'comments': ['I love this!']})
        self.assertEqual(response.status_code, 200)

        json_data = response.get_json()
        self.assertIsInstance(json_data, list)
        self.assertGreater(len(json_data), 0)

        # Each comment should have a 'sentiment' field in [-1, 0, 1]
        for item in json_data:
            self.assertIn('comment', item)
            self.assertIn('sentiment', item)
            self.assertIn(item['sentiment'], [-1, 0, 1])

if __name__ == '__main__':
    unittest.main()
