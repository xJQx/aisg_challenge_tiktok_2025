import json
from typing import Callable, Any
from datasets import load_dataset
from src.config import OUTPUT_DIR, SKIP_PROCESSED_VIDEOS, VLLM_API_URL
from src.utils.call_mistral_model import call_mistral_vllm
from src.utils.downloader import VideoDownloader
from src.utils.frame_extractor import FrameExtractor
from src.annotator import (
    FrameAnnotator, VideoAnnotator,
    AnnotationSummarizer
)

class SubQuestionGenerator:
    def __init__(self, call_model: Callable[..., Any], vllm_url):
        self.call_model = call_model
        self.vllm_url = vllm_url

    def generate(self, main_question: str) -> Any:
        print("\tGenerating sub-questions...")
        prompt = (
            f"Given the main question: '{main_question}', generate sub-questions to better understand the video. Subquestions generated should be adversarial in nature, to ensure robustness of the model response to the main question."
        )
        return self.call_model(self.vllm_url, prompt)


class VideoProcessor:
    def __init__(self, call_model, vllm_url):
        self.downloader = VideoDownloader()
        self.extractor = FrameExtractor()
        self.frame_annotator = FrameAnnotator(call_model, vllm_url=vllm_url)
        self.video_annotator = VideoAnnotator(call_model, vllm_url=vllm_url)
        self.summarizer = AnnotationSummarizer(call_model, vllm_url=vllm_url)
        self.subq_gen = SubQuestionGenerator(call_model, vllm_url=vllm_url)
        self.vllm_url = vllm_url

    def process(self, example: dict):
        qid = example["qid"]
        vid = example["video_id"]
        out_path = OUTPUT_DIR / f"{qid}_{vid}.json"
        print(f"\nProcessing {qid}, video {vid}…")

        if SKIP_PROCESSED_VIDEOS and out_path.exists():
            print(f"\t✅ Already done: {out_path}")
            return

        # 0. Download
        video_path = self.downloader.download(
            example["youtube_url"], qid, vid
        )

        # 1. Sub-questions
        subqs = self.subq_gen.generate(example["question"])

        # 2. Extract frames
        frames, timestamps = self.extractor.extract(video_path)

        # 3a. Frame-level annotations
        frame_anns = self.frame_annotator.annotate(
            frames, timestamps, example["question"], subqs, qid, vid
        )

        # 3b. Video-level annotation
        whole_ann = self.video_annotator.annotate(
            example["question"], subqs, frame_anns
        )

        # 4. Summarize
        summary = self.summarizer.summarize(frame_anns, whole_ann)

        # 5. Save everything
        result = {
            "qid": qid,
            "video_id": vid,
            "video_path": video_path,
            "main_question": example["question"],
            "sub_questions": subqs,
            "annotations": {
                "frame_annotations": frame_anns,
                "whole_video_annotation": whole_ann,
                "annotations_summary": summary
            }
        }
        self._save_result(result, out_path)

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

    # Call Model Function
    print("Using vLLM Mistral Model")

    # Video Processor
    print("[Video Processing]")
    for example in dataset:
        VideoProcessor(call_mistral_vllm, VLLM_API_URL).process(example)
        break