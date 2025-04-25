import json
from datetime import datetime, timedelta
from typing import Callable, Any
import pandas as pd
import sys
import math

from datasets import load_dataset
from src.config import OUTPUT_DIR, SKIP_PROCESSED_VIDEOS,\
      VLLM_API_URL_1, VLLM_API_URL_2, VLLM_API_URL_3, VLLM_API_URL_4,\
      VLLM_API_URL_5, VLLM_API_URL_6, VLLM_API_URL_7, VLLM_API_URL_8, \
      VLLM_API_URL_9
from src.utils.call_mistral_model import call_mistral_vllm
from src.utils.downloader import VideoDownloader
from src.utils.frame_extractor import FrameExtractor
from src.annotator import (
    FrameAnnotator, VideoAnnotator,
    AnnotationSummarizer
)

class VideoKeyFramesProcessor:
    def __init__(self, call_model, vllm_url):
        self.frame_annotator = FrameAnnotator(call_model, processBlob=True, vllm_url=vllm_url)

    def process(self, example: dict, example_keyframes_data, example_current_result: dict, out_path: str):
        qid, vid = example["qid"], example["video_id"]
        print(f"\nProcessing {qid}, video {vid}…")

        # Extract timestamps and frames
        timestamps = []
        keyframes = []
        for _, row in example_keyframes_data.iterrows():
            timestamps.append(row["timestamp"])
            keyframes.append(row["frame"])
        
        # 1. KeyFrames-level annotations
        keyframes_anns = self.frame_annotator.annotate(
            keyframes, timestamps, example["question"], example_current_result["sub_questions"], qid, vid
        )

        # 2. Save everything
        if keyframes_anns:
            example_current_result["annotations"]["keyframes_annotations"] = keyframes_anns
            self._save_result(example_current_result, out_path)

    @staticmethod
    def _save_result(data: dict, path):
        """
        Ensure the output directory exists, write `data` as pretty JSON
        to `path`, and log completion.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Done: {path}")
    
    @staticmethod
    def _map_server_number_to_vllm_url(server_number: str):
        if server_number == "1":
            return VLLM_API_URL_1
        if server_number == "2":
            return VLLM_API_URL_2
        if server_number == "3":
            return VLLM_API_URL_3
        if server_number == "4":
            return VLLM_API_URL_4
        if server_number == "5":
            return VLLM_API_URL_5
        if server_number == "6":
            return VLLM_API_URL_6
        if server_number == "7":
            return VLLM_API_URL_7
        if server_number == "8":
            return VLLM_API_URL_8
        if server_number == "9":
            return VLLM_API_URL_9
        else:
            raise Exception("Invalid server number")


if __name__ == "__main__":
    # Dataset
    print("Loading Dataset...")
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    print(dataset)

    server_number = sys.argv[2] if len(sys.argv) > 2 else 1
    print(f"Using Server {server_number}")

    # Split into chunks of 60
    total_examples = len(dataset)
    chunk_size = 60
    num_chunks = math.ceil(total_examples / chunk_size)

    # Select which chunk to run (default = 0)
    CHUNK_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if CHUNK_ID < 0 or CHUNK_ID >= num_chunks:
        print(f"Invalid CHUNK_ID: {CHUNK_ID}. Should be 0 to {num_chunks - 1}.")
        sys.exit(1)

    start_idx = CHUNK_ID * chunk_size
    end_idx = min(start_idx + chunk_size, total_examples)
    selected_dataset = dataset.select(range(start_idx, end_idx))
    print(f"Processing chunk {CHUNK_ID} with {len(selected_dataset)} examples [{start_idx}:{end_idx}]")

    # Load Key Frames Dataset
    print("Loading Key Frames Dataset...")
    combined_foi_data = pd.read_parquet("hf://datasets/lemousehunter/combined_foi/final_combined_foi.parquet")
    # combined_foi_data = pd.read_parquet("data/final_combined_foi.parquet")

    # Call Model Function
    print("Using vLLM Mistral Model")

    # VideoKeyFramesProcessor
    print("[Video KeyFrames Processor]")
    for example in selected_dataset:
        qid, vid = example["qid"], example["video_id"]

        # Check if there's a output.json file for his qid
        out_path = OUTPUT_DIR / f"{qid}_{vid}.json"
        if not out_path.exists():
            print(f"\tSkipping... Output file does not exist: {out_path}")
            continue
        
        with open(out_path, "r") as f:
            example_current_result = json.load(f)

        # Check if keyframes dataset contain the correct qid and vid
        example_keyframes_data = combined_foi_data[combined_foi_data["qid"] == qid]
        if not example_keyframes_data.empty:
            example_keyframes_data = example_keyframes_data.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        else:
            print("Skipping... No keyframes exist.")
            continue

        VideoKeyFramesProcessor(call_mistral_vllm, VideoKeyFramesProcessor._map_server_number_to_vllm_url(server_number)).process(
            example, 
            example_keyframes_data, 
            example_current_result,
            out_path)
    print("BATCH DONE! 🚀🚀🚀")
