import json
import math
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Callable, Any
from src.utils.frame_encoder import encode_to_base64, encode_blob_to_base64
from src.config import ERROR_DIR


class FrameAnnotator:
    def __init__(self, call_model: Callable[..., Any], batch_size: int = 10, processBlob: bool = False, vllm_url = ""):
        self.call_model = call_model
        self.batch_size = batch_size
        self.processBlob = processBlob
        self.vllm_url = vllm_url
        ERROR_DIR.mkdir(parents=True, exist_ok=True)

    def annotate(
        self,
        frames: List,
        timestamps: List[float],
        main_question: str,
        sub_questions,
        question_id: str,
        video_id: str
    ) -> List[dict]:
        print("\tAnnotating Extracted Frames...")
        annotations: List[dict] = []

        for i in range(0, len(frames), int(self.batch_size)):
            print(f"\t\tProcessing batch {i}...")
            batch_f = frames[i : i + self.batch_size]
            batch_ts = timestamps[i : i + self.batch_size]
            imgs_b64 = [encode_to_base64(f) if not self.processBlob else encode_blob_to_base64(f) for f in batch_f]
            previous = annotations[-1]["annotation"] if annotations else None

            prompt = self._build_prompt(batch_ts, main_question, sub_questions, previous)
            try:
                print('Frame annotator posting to', self.vllm_url)
                raw = self.call_model(self.vllm_url, prompt, imgs_b64)
                parsed = self._parse_response(raw)
                annotations.extend(parsed)
            except Exception as e:
                self._write_error(e, question_id, video_id)
        return annotations

    def _build_prompt(self, batch_ts, main_question, sub_questions, previous) -> str:
        p = (
            f"Instruction: You are shown {len(batch_ts)} frames from a video. Briefly describe each, "
            "noting changes from the previous frame, specifically noting changes in entity state, position, appearance and existence based on its descriptive text. If the entity is a person, describe the gender and clothes of the person. If it is an item, describe the type and function of item. Do not hallucinate. Do not state anything you are unsure of. Only answer questions you are very sure about. Firstly, identify entities in the video, and give them descriptive texts for future references. Also identify interactions between entities. Next, answer the Subquestions. Identify the relevance of entities found with respect to the subquestions. Next, using the sub-question answers, answer the main-question. Then use your answers for the Subquestions to formulate your annotation for the frame, bearing in mind it will late be used for answering the main question eventually."
            f"User main question: \"{main_question}\"\n"
            f"Subquestions: {sub_questions}"
        )
        for idx, ts in enumerate(batch_ts):
            p += f"Frame {idx}: {ts:.2f}s. Previous: {previous}\n"
        p += (
            "\nReturn as a JSON array: "
            "[{\"timestamp\":0.0,\"annotation\":\"...\"}, ...]"
        )
        return p

    @staticmethod
    def _parse_response(raw: str) -> List[dict]:
        txt = raw.strip("```").replace("json", "")
        start = txt.find("[")
        end = txt.rfind("]") + 1
        if start < 0 or end < 0:
            raise ValueError("Invalid response format")
        return json.loads(txt[start:end])

    @staticmethod
    def _write_error(err: Exception, qid: str, vid: str):
        fp = ERROR_DIR / f"{qid}_{vid}.json"
        with open(fp, "w") as f:
            json.dump({"error": str(err)}, f, indent=2)