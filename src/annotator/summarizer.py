from typing import Callable, Any, List


class AnnotationSummarizer:
    def __init__(self, call_model: Callable[..., Any]):
        self.call_model = call_model

    def summarize(self, frame_annotations: List[dict], whole_annotation: str) -> str:
        print("\tSummarizing all annotations...")
        lines = "\n".join(
            f"[{a['timestamp']:.1f}s]: {a['annotation']}"
            for a in frame_annotations
        )
        prompt = (
            "Frame-level descriptions:\n"
            f"{lines}\n\n"
            "Video-level description:\n"
            f"{whole_annotation}\n\n"
            "Write a brief summary."
        )
        return self.call_model(prompt)