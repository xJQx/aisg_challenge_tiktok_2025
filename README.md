**Tables of Content**

- [SSH and Vast.ai](#ssh-and-vastai)
  - [✅ Option 1: If You Already Have an SSH Key](#-option-1-if-you-already-have-an-ssh-key)
  - [✅ Option 2: If You Don’t Have a Key Yet](#-option-2-if-you-dont-have-a-key-yet)
  - [SSH into Vast.ai](#ssh-into-vastai)
- [Setting up GPU Virtual Server](#setting-up-gpu-virtual-server)

---

# SSH and Vast.ai

**Vast.ai** is a great, cost-efficient option for spinning up GPU servers.

Here’s how you can get your **SSH public key** and provide it to your tech lead:

## ✅ Option 1: If You Already Have an SSH Key

Check if you already have a key:

```bash
ls ~/.ssh/id_rsa.pub
```

If you see the file, show the contents with:

```bash
cat ~/.ssh/id_rsa.pub
```

Copy the whole output and give that to your tech lead.

## ✅ Option 2: If You Don’t Have a Key Yet

Generate a new SSH key pair:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Just press **Enter** through the prompts unless you want to set a passphrase.

Then check:

```bash
cat ~/.ssh/id_rsa.pub
```

That prints your **public key** — it looks like this:

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQ... your_email@example.com
```

Give that full string to your tech lead — **not the private key (`id_rsa`)!**

## SSH into Vast.ai

```bash
ssh -i ~/.ssh/id_rsa -p 13642 root@80.188.223.202
```

---

# Setting up GPU Virtual Server

Step-by-Step: Get Your Environment Ready (if it has not been set up)

Let’s set up:

✅ Python + virtualenv

✅ Required packages (Torch, Transformers, vLLM, etc.)

✅ Optional: tmux so your server can keep running in the background

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
pip install transformers accelerate sentencepiece datasets

# vLLM
pip install vllm
```

✅ If you're not sure your CUDA version: nvidia-smi
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

If not, let me know and we’ll debug the CUDA/driver setup.

🧠 4. Run Mistral with vLLM
You can now launch the OpenAI-compatible vLLM server like this:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer_mode mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --limit_mm_per_prompt 'image=10' \
  --gpu-memory-utilization 0.95 \
  --swap-space 16
  --max-model-len 12000



# OR a smaller model
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --host 0.0.0.0 --port 8000 --tensor-parallel-size 4
```

💡 5. (Optional) Run in tmux so it stays alive

```bash
tmux new -s vllm

# Start your vLLM server here
```

To detach: `Ctrl + B`, then press `D`
To resume later: `tmux attach -t vllm`

✅ All Set
You can now:

Call the model from your local machine via the OpenAI API interface (just change the port/IP)

Use `call_mistral()` in your script to hit `http://<your-ip>:40311/v1/completions`

Process frames, annotate, summarize, etc. — now fast with GPU

---

ssh -i ~/.ssh/id_rsa -p 13642 root@80.188.223.202

source venv/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
 --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
 --host 0.0.0.0 \
 --port 8000 \
 --tokenizer_mode mistral \
 --tool-call-parser mistral \
 --enable-auto-tool-choice \
 --limit_mm_per_prompt 'image=10' \
 --gpu-memory-utilization 0.95 \
 --swap-space 16 \
--max-model-len 12000

python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --host 0.0.0.0 --port 8000 --tokenizer_mode mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --gpu-memory-utilization 0.95 --swap-space 16 --max-model-len 12000

---

pkill -f vllm
rm -rf models/
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503

---

default max_model_len = 128,000
works

- 12,000
- 1,024

doesn't work

- 128,000
- 64,000
- 48,000
- 24,000

---

GH200 (https://docs.google.com/document/d/1RrWEsjxh7K1VYTZGEfIuB5pwSrXaXfX90o8Tp312Y-M/edit?tab=t.0)
ssh -i ~/.ssh/lambda-dev.cer ubuntu@192.222.51.193

```bash
cd llama.cpp &&

pm2 start -n "llama.cpp" "./build/bin/llama-server --model ../Mistral-Small-3.1-24B-Instruct-2503-Q8_0-GGUF/mistral-small-3.1-24b-instruct-2503-q8_0.gguf --threads 1 --port 7859 --n-gpu-layers 100 --ctx-size 50000 --parallel 5 --cont-batching --host 0.0.0.0"
```

pm2 logs
pm2 stop llama.cpp
pm2 delete llama.cpp
pm2 list
pm2 restart llama.cpp # Restart the server
