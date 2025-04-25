import json
import requests

from src.config import QWEN_GH200_API_URL


def __load_qwen2_system_prompt():
    return f"""
You are Qwen2-72B-Instruct, a highly capable assistant trained by Alibaba on a wide range of tasks. You have extensive reasoning ability.

# RULES
- Do not make up information.
- Respond concisely but informatively.
- Always use the language of the user's question.
- If a format is provided, follow it strictly.
    """

def __buildQwen2Messages(prompt: str) -> list:
    system_prompt = __load_qwen2_system_prompt()
    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": prompt }
    ]
    return messages


def call_qwen2_model(prompt: str):
    try:
        model = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
        messages = __buildQwen2Messages(prompt)

        data = {
            "model": model,
            # "prompt": messages[0]['content'] + messages[1]['content']
            "messages": messages,
            "temperature": 0.2,
            # "max_tokens": 1024
        }

        headers = {
            "Content-Type": "application/json"
        }
        
        print("\tMaking request to GH200")
        response = requests.post(QWEN_GH200_API_URL, headers=headers, data=json.dumps(data))
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"❌ Error calling Qwen2 GH200 server: {e}")
        return "ERROR"
