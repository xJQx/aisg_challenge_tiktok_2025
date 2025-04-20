from datasets import load_dataset

from scripts.phase1_process import phase1_process_video
from scripts.models import MistralModel_Local, call_mistral_vllm
from scripts.config import USE_LOCAL_MODEL

if __name__ == "__main__":
    # Dataset
    print("Loading Dataset...")
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    print(dataset)
    
    # Call Model Function
    if (USE_LOCAL_MODEL):
        print("Loading Local Model...")
        call_model = MistralModel_Local().call_mistral
    else:
        print("Using vLLM Mistral Model")
        call_model = call_mistral_vllm

    print("[Phase 1 Processing]")
    for idx, example in enumerate(dataset.select(range(2))): # Limit to 2 for now
        phase1_process_video(example, call_model)

        break