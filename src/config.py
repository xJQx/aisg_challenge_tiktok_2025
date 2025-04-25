from pathlib import Path

# Settings
FRAME_INTERVAL_SECONDS = 1 # seconds
SKIP_PROCESSED_VIDEOS = False

# Outputs Directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Outputs Directory
SUBMISSION_DIR = Path("submissions")
SUBMISSION_DIR.mkdir(exist_ok=True)

# Error Directory
ERROR_DIR = Path("error")
ERROR_DIR.mkdir(exist_ok=True)

# YouTube Downloads Directory
VIDEO_DOWNLOAD_DIR = Path("videos")
VIDEO_DOWNLOAD_DIR.mkdir(exist_ok=True)

# VLLM Settings
USE_GH200 = False
VLLM_API_URL = "http://192.222.51.193:7859/v1/chat/completions" if USE_GH200 else "http://80.188.223.202:13301/v1/chat/completions" 
# QWEN_GH200_API_URL = "http://192.222.51.193:7859/v1/chat/completions"
QWEN_GH200_API_URL = "http://198.145.126.235:21527/v1/chat/completions"

VLLM_API_URL_1 = "http://80.188.223.202:13301/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 13642 root@80.188.223.202
VLLM_API_URL_2 = "http://115.124.123.240:19039/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 30540 root@115.124.123.240
VLLM_API_URL_3 = "http://198.145.126.233:39068/v1/chat/completions" # x2 ssh -i ~/.ssh/id_rsa -p 32617 root@198.145.126.233
VLLM_API_URL_5 = "http://198.145.126.235:33183/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 16998 root@198.145.126.235
VLLM_API_URL_6 = "http://198.145.126.233:22370/v1/chat/completions" # x2 ssh -i ~/.ssh/id_rsa -p 28822 root@198.145.126.233
VLLM_API_URL_7 = "http://198.145.126.235:24425/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 25293 root@198.145.126.235
VLLM_API_URL_8 = "http://198.145.126.239:40157/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 40059 root@198.145.126.239
VLLM_API_URL_9 = "http://198.145.126.236:25429/v1/chat/completions" # ssh -i ~/.ssh/id_rsa -p 20794 root@198.145.126.236
VLLM_API_URL_4 = "http://198.145.126.232:31242/v1/chat/completions" #2 ssh -i ~/.ssh/id_rsa -p 39300 root@198.145.126.232