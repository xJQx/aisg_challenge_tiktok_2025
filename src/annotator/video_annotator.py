from typing import List, Callable, Any



class VideoAnnotator:
    def __init__(self, call_model: Callable[..., Any]):
        self.call_model = call_model

    def annotate(self, main_question: str, frame_annotations: List[dict]) -> str:
        print("\tAnnotating video as a whole...")
        content = "\n".join(
            f"[Frame {i} ({ann['timestamp']:.2f}s)] {ann['annotation']}"
            for i, ann in enumerate(frame_annotations)
        )
        prompt = (
            f"Given these frame annotations:\n{content}\n\n"
            f"User question: \"{main_question}\"\n"
            "Provide a short overall description of the video."
        )
        return self.call_model(prompt)