from pathlib import Path

# Settings
FRAME_INTERVAL_SECONDS = 1/24 # seconds
SKIP_PROCESSED_VIDEOS = True

# Outputs Directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Error Directory
ERROR_DIR = Path("error")
ERROR_DIR.mkdir(exist_ok=True)

# YouTube Downloads Directory
VIDEO_DOWNLOAD_DIR = Path("videos")
VIDEO_DOWNLOAD_DIR.mkdir(exist_ok=True)

# VLLM Settings
USE_GH200 = False
VLLM_API_URL = "http://192.222.51.193:7859/v1/chat/completions" if USE_GH200 else "http://80.188.223.202:13301/v1/chat/completions" 