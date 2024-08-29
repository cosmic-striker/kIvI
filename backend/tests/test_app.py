import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_chat_no_input(self):
        response = self.app.post('/chat', json={})
        self.assertEqual(response.status_code, 400)

    def test_chat_with_input(self):
        response = self.app.post('/chat', json={'message': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json)

if __name__ == '__main__':
    unittest.main()
