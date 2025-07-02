import openai
from .base import EmbeddingModel


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, model, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]
