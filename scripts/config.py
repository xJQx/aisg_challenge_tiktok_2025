from pathlib import Path

# Settings
FRAME_INTERVAL_SECONDS = 1 # seconds

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
VLLM_API_URL = "http://80.188.223.202:13301/v1/chat/completions"