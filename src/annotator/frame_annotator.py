import json
from typing import List, Callable, Any
from src.utils.frame_encoder import encode_to_base64, encode_blob_to_base64
from src.config import ERROR_DIR


class FrameAnnotator:
    def __init__(self, call_model: Callable[..., Any], batch_size: int = 10, processBlob: bool = False):
        self.call_model = call_model
        self.batch_size = batch_size
        self.processBlob = processBlob
        ERROR_DIR.mkdir(parents=True, exist_ok=True)

    def annotate(
        self,
        frames: List,
        timestamps: List[float],
        main_question: str,
        question_id: str,
        video_id: str
    ) -> List[dict]:
        print("\tAnnotating Extracted Frames...")
        annotations: List[dict] = []

        for i in range(0, len(frames), self.batch_size):
            print(f"\t\tProcessing batch {i}...")
            batch_f = frames[i : i + self.batch_size]
            batch_ts = timestamps[i : i + self.batch_size]
            imgs_b64 = [encode_to_base64(f) if not self.processBlob else encode_blob_to_base64(f) for f in batch_f]
            previous = annotations[-1]["annotation"] if annotations else None

            prompt = self._build_prompt(batch_ts, main_question, previous)
            try:
                raw = self.call_model(prompt, imgs_b64)
                parsed = self._parse_response(raw)
                annotations.extend(parsed)
            except Exception as e:
                self._write_error(e, question_id, video_id)
        return annotations

    def _build_prompt(self, batch_ts, main_question, previous) -> str:
        p = (
            f"You are shown {len(batch_ts)} frames from a video. Briefly describe each, "
            "noting changes from the previous frame.\n"
            f"User question: \"{main_question}\"\n\n"
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