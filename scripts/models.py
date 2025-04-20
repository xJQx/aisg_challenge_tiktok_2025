import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from scripts.config import VLLM_API_URL

def call_mistral_vllm(prompt, max_tokens=300):
    try:
        response = requests.post(
            VLLM_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                # "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "c": 0.15
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Error calling vLLM server: {e}")
        return "ERROR"

class MistralModel_Local:
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cpu"

    def __init__(self):
        # Setup model and tokenizer
        print("🔄 Loading Mistral model (CPU mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)

    # Mistral prompt caller
    def call_mistral(self, prompt: str, max_tokens: int = 256) -> str:
        print(f"\t\tcalling model {self.model_id}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.15,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
