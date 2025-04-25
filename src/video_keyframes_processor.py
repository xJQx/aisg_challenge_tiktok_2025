import json
from datetime import datetime, timedelta
from typing import Callable, Any
import pandas as pd

from datasets import load_dataset
from src.config import OUTPUT_DIR, SKIP_PROCESSED_VIDEOS
from src.utils.call_mistral_model import call_mistral_vllm
from src.utils.downloader import VideoDownloader
from src.utils.frame_extractor import FrameExtractor
from src.annotator import (
    FrameAnnotator, VideoAnnotator,
    AnnotationSummarizer
)

class VideoKeyFramesProcessor:
    def __init__(self, call_model):
        self.frame_annotator = FrameAnnotator(call_model, processBlob=True)
        self.video_annotator = VideoAnnotator(call_model)
        self.summarizer = AnnotationSummarizer(call_model)

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



if __name__ == "__main__":
    # Dataset
    print("Loading Dataset...")
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    print(dataset)

    # Load Key Frames Dataset
    print("Loading Key Frames Dataset...")
    # combined_foi_data = pd.read_parquet("hf://datasets/lemousehunter/combined_foi/final_combined_foi.parquet")
    combined_foi_data = pd.read_parquet("data/final_combined_foi.parquet")

    # Call Model Function
    print("Using vLLM Mistral Model")

    # VideoKeyFramesProcessor
    print("[Video KeyFrames Processor]")
    for example in dataset:
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

        VideoKeyFramesProcessor(call_mistral_vllm).process(
            example, 
            example_keyframes_data, 
            example_current_result,
            out_path)
        
        break