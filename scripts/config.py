from pathlib import Path

# Settings
FRAME_INTERVAL_SECONDS = 1 # seconds

# Outputs Directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# YouTube Downloads Directory
VIDEO_DOWNLOAD_DIR = Path("videos")
VIDEO_DOWNLOAD_DIR.mkdir(exist_ok=True)

# VLLM Settings
VLLM_API_URL = "http://173.49.133.105:16920/v1/completions"
USE_LOCAL_MODEL = False