from datasets import load_dataset

from scripts.phase1_process import phase1_process_video
from scripts.models import call_mistral_vllm

if __name__ == "__main__":
    # Dataset
    print("Loading Dataset...")
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    print(dataset)
    
    # Call Model Function
    print("Using vLLM Mistral Model")
    call_model = call_mistral_vllm

    print("[Phase 1 Processing]")
    for idx, example in enumerate(dataset):
        phase1_process_video(example, call_model)
