import requests
import json
from datasets import load_dataset

from scripts.models import __buildMessages
from scripts.phase1_process import _download_youtube_video, _extract_frames, _annotate_frames
from scripts.models import call_mistral_vllm

dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
print(dataset)

for idx, example in enumerate(dataset):
  video_id = example['video_id']
  question_id = example['qid']
  video_youtube_url = example["youtube_url"]
  main_question = example["question"]
  question_prompt = example["question_prompt"]

  # 0. Download Youtube Video to local if doesn't exist
  video_path = _download_youtube_video(video_youtube_url, question_id, video_id)

  # 1. Extract Frames
  frames, timestamps = _extract_frames(video_path, interval=1)

  # 2a. Annotate Extracted Frames
  frame_annotations = _annotate_frames(frames, timestamps, main_question, call_mistral_vllm, question_id, video_id)
  break

# prompt = "Which region is Singapore located in?"

# model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
# messages = __buildMessages(model, prompt, None)

# data = { "model": model, "messages": messages, "temperature": 0.15 } # If want to limit response: "max_tokens": 128
# headers = { "Content-Type": "application/json", "Authorization": "Bearer token" }
# response = requests.post(
#     "http://192.222.51.193:7859/v1/chat/completions", 
#     headers=headers, 
#     data=json.dumps(data)
# )
# print(response.json())
# print(response.json()["choices"][0]["message"]["content"])
