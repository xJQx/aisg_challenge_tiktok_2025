import json
from datetime import datetime, timedelta

import requests
from huggingface_hub import hf_hub_download

from src.config import VLLM_API_URL


def __load_system_prompt(repo_id: str, filename: str, useDefault: bool = False) -> str:
    if useDefault:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(file_path, "r") as file:
            system_prompt = file.read()
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        model_name = repo_id.split("/")[-1]
        return system_prompt.format(name=model_name, today=today, yesterday=yesterday)
    else:
        system_prompt = """
You are Mistral-Small-3.1-24B-Instruct-2503, a Large Language Model (LLM).

When you're not sure about some information, don't make up anything.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, 
you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.
"""
        return system_prompt

def __buildMessages(model, prompt, image_base64_list: list):
    SYSTEM_PROMPT = __load_system_prompt(model, "SYSTEM_PROMPT.txt", useDefault=False)
    if image_base64_list:
        content = [
            { "type": "text", "text": prompt }
        ]
        for image_base64 in image_base64_list:
            content.append({ "type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"} })
    else:
        content = [{ "type": "text", "text": prompt }]

    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    return messages

def call_mistral_vllm(url: str, prompt, image_base64=None):
    try:
        model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        messages = __buildMessages(model, prompt, image_base64)

        data = { "model": model, "messages": messages, "temperature": 0.15 } # If want to limit response: "max_tokens": 128
        headers = { "Content-Type": "application/json", "Authorization": "Bearer token" }
        print('posting to url:', url)
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data)
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Error calling vLLM server: {e}")
        return "ERROR"