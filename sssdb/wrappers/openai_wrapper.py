import numpy as np


class OpenAIWrapper:
    import openai

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        openai.api_key = api_key
        self.model = model

    def encode(self, text: str) -> np.ndarray:
        response = openai.Embedding.create(
            model=self.model,
            input=text,
        )
        return [np.array(item["embedding"]) for item in response["data"]]
