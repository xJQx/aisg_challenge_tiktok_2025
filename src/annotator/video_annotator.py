from typing import List, Callable, Any



class VideoAnnotator:
    def __init__(self, call_model: Callable[..., Any], vllm_url):
        self.call_model = call_model
        self.vllm_url = vllm_url

    def annotate(self, main_question: str, subquestions: str, frame_annotations: List[dict]) -> str:
        print("\tAnnotating video as a whole...")
        content = "\n".join(
            f"[Frame {i} ({ann['timestamp']:.2f}s)] {ann['annotation']}"
            for i, ann in enumerate(frame_annotations)
        )
        prompt = (
            f"Instruction: You are given the frame annotations of a video, along with subquestions and the main question. Use them to write a summary of the video. This summary should answer the sub-questions and main question. Identify entities (humans, objects, items) that may be relevant to the subquestions / main question."
            f"Frame Annotations:\n{content}\n\n"
            f"User question: \"{main_question}\"\n"
            f"Sub-questions: {subquestions}\n"
        )
        return self.call_model(self.vllm_url, prompt)