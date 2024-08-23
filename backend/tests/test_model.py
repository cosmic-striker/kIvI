import unittest
from model.llama2_model import LLaMA2Model

class LLaMA2ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = LLaMA2Model()

    def test_generate_response(self):
        prompt = "Hello, how are you?"
        response = self.model.generate_response(prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()
