import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLaMA2Model:
    def __init__(self, model_name="llama2", model_path="model/llama2/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """Generates a response from the LLaMA 2 model based on the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def get_response(self, user_input):
        """Processes user input and returns a response."""
        return self.generate_response(user_input)
