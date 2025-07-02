
import openai
from .base import LLM

class OpenAILLM(LLM):
    def __init__(self, model, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
