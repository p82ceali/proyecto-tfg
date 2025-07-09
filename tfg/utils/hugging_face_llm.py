# utils/huggingface_llm.py

from transformers import pipeline
import os

class HuggingFaceLLM:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
            token=os.getenv("HF_API_TOKEN"),
            max_new_tokens=300,
            do_sample=True
        )

    def call(self, prompt: str) -> str:
        result = self.generator(prompt, return_full_text=False)[0]["generated_text"]
        return result.strip()
