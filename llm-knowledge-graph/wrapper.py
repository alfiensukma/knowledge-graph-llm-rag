import requests
from langchain.schema import BaseLLM, LLMResult

class ChatLMStudio(BaseLLM):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def _generate(self, prompt: str, **kwargs) -> LLMResult:
        response = requests.post(
            f"{self.api_url}/completions",
            json={"prompt": prompt, **kwargs}
        )
        response.raise_for_status()
        return LLMResult(generations=[response.json()["choices"][0]["text"]])

    def _llm_type(self) -> str:
        return "lm-studio"