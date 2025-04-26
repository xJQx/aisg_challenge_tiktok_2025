# Singapore AI Student Challenge 2025 - TikTok Track

**Challenge Link:** https://developers.tiktok.com/ai/hackathon

**Tables of Content**

- [Singapore AI Student Challenge 2025 - TikTok Track](#singapore-ai-student-challenge-2025---tiktok-track)
  -  [Challenge Background](challenge-background)
- [Architecture](#architecture)
- [Setting Up](#setting-up)
  - [Prepare GPU server connection url](#prepare-gpu-server-connection-url)
  - [Setting up GPU Server](#setting-up-gpu-server)
  - [Set up another GPU server for a secondary model](#set-up-another-gpu-server-for-a-secondary-model)
- [Solution Pipeline](#solution-pipeline)
  - [1. Video Processor](#1-video-processor)
  - [2. Video Vectorizer](#2-video-vectorizer)
  - [3. Key Frames Processor](#3-key-frames-processor)
  - [4. Video Answering](#4-video-answering)



## Challenge Background

Although the capabilities of Multimodal Large Language Models (MLLMs) are continuously improving, we find that there is still a gap between models and humans in terms of cognitive understanding. We tested the differences in cognitive understanding between models and humans, then classified, and analyzed them. The results are as follows:

For an objective fact, humans do not deviate in their cognition just because the fact is described in different ways. Human understanding of an objective fact has a certain degree of consistency, even under rigorous testing.

We found that artificial intelligence models do not possess such consistency in many scenarios. This problem is reflected in our testing and is categorized as follows:


i. Models may correctly answer a single question but fail when the same fact is queried differently.

ii. Models may give a wrong answer to a single question and still fail when the question is rephrased, finding an alternate wrong answer to the same objective fact.


iii. Models may stay consistent through their judgement, but the initial judgement is simply incorrect compared to the objective truth.

We hope that hosting this event can inspire more innovative solutions to address deficiencies in the capability of consistent cognitive understanding



# Architecture

![aisg_challenge_2025_tiktok](https://github.com/user-attachments/assets/d85a6724-d9a5-4c97-9f1d-c3417bc49f1f)


# Setting Up

## Prepare GPU server connection url

Prepare a GPU server connection url. Preferably a A100/H100 GPU with high computational capability.

## Setting up GPU Server

In the GPU server of your choice, please setup the GPU Server environment by following the steps outlined in this section.

📦 1. System & Python Setup

```bash
sudo apt update && sudo apt install -y python3-venv git wget curl tmux
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

🔥 2. Install CUDA-optimized PyTorch + vLLM stack

```bash
pip install --upgrade pip

# Install PyTorch with GPU (check CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face stack
pip install transformers accelerate sentencepiece datasets vllm

# vLLM
pip install vllm
```

✅ If you're not sure your CUDA version: `nvidia-smi`
Use cu118 for CUDA 11.8 or cu121 for CUDA 12.1

🧪 3. Verify GPU is available to PyTorch
Run this:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

You should get:

```python
True
```

🧠 4. Run Mistral with vLLM
You can now launch the OpenAI-compatible vLLM server like this:

```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 1 --gpu-memory-utilization 0.99 --swap-space 16 --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 12000 --max-num-seqs=10
```

✅ All Set
You can now:

Call the model from your local CPU machine via the OpenAI API interface by hitting the `http://<your-ip>:<your-port>/v1/chat/completions` endpoint.

## Set up another GPU server for a secondary model

Follow the above steps for a 2nd GPU server, but serve the following model instead.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 --max-model-len 12000 --gpu-memory-utilization 0.99 --dtype bfloat16 --tensor-parallel-size 1 --swap-space 16 --host 0.0.0.0 --port 8000
```

# Solution Pipeline

## 1. Video Processor

Reference: `/src/video_processor.py`
Model Used: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`

Note: Please set the `VLLM_API_URL` value in the `src/config.py`

In this step, we:

1. Download the benchmark dataset
2. Generate 3 subquestions based on the main question provided in the benchmark dataset
3. Naively extract Frames from the video (1 frame per second)
4. Annotate all the extract frames (frame-level annotation) by passing in the image frames and the question into the model.
5. Annotate the video on the whole video level
6. Summarize all the annotations
7. Save the generated results into a json file for each question in the `outputs` directory.
8. ssh -i ~/.ssh/id_rsa -p 13642 root@80.188.223.202

To run the Video Processor, run the following in the root directory:

```bash
python src/video_processor.py
```

## 2. Video Vectorizer

Reference: `/src/video_vectorizer.py`
Model Used: `jina-ai/clip-as-service`

Note: Please set the `VLLM_API_URL` value in the `src/config.py`

In this step, we:

1. Download the videos
2. Extract timestamps at a higher rate of 24 frames per second
3. Build a DocumentArray for frames and Vectorize frames via Jina encoder
4. Vectorize question as a Document with text
5. Find similar frames above threshold
6. Collect records from results
7. Save to Parquet via Hugging Face Dataset or write error

To run the Video Vectorizer, run the following in the root directory:

```bash
python src/video_vectorizer.py
```

## 3. Key Frames Processor

Reference: `/src/video_keyframes_processor.py`
Model Used: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`

Note: Please set the `VLLM_API_URL` value in the `src/config.py`

In this step, we:

1. For each keyframe generated in Step 2 (Video Vectorizer) above, we pass the keyframe and its corresponding timestamp into a model to annotate them further. Supplementing the original annotations generated in step 1 with "keyframes_annotations"

To run the Video Key Frames Processor, run the following in the root directory:

```bash
python src/video_keyframes_processor.py <batch_number> <server_number>
```

Note:

- batch_number: ranges from 0-24
- server_number: ranges from 1-9 (depending on the number of GPU servers you have in `src/config.py`) (for parallel processing)

## 4. Video Answering

Reference: `/src/video_answering.py`
Model Used: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`

Note: Please set the `QWEN_GH200_API_URL` value in the `src/config.py`

This step is the question and answering part of our model pipeline.

It handles Question & Answering using a 3-stage process:

- 0-shot: Determine question type (MCQ or Open-Ended)
- 1-shot: Validate Open-Ended questions for contextual fit
- 2-shot: Answer based on question type

0 Shot: figure type of question (open ended or mcq)

- Is the question Open-Ended or MCQ? If it is Open-Ended, answer 'OE'. If it is MCQ, answer it as 'MCQ'.

1 Shot: (Ask only if it is OE): Does the question makes sense in relation to the context given

- Does it make sense? Does what the qns ask for exist in the video? If not, what is the most relevant entity that exists in the video instead.

2 Shot:

- If MCQ: State your answer and explain in a step-by step manner. Follow the given format strictly when responding:
  ANSWER: {OPTION}
  EXPLAINATION:\n{EXPLANATION}
- If OE: First, answer the sub-questions. Then, use your answer for the sub-questions to answer the main-question.

To run the Video Answering, run the following in the root directory:

```bash
python src/video_answering.py
```
