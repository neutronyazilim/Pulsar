from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "pulsar-coder-1.5b" / "final"
MERGED_PATH = ROOT / "models" / "pulsar-coder-1.5b-merged"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"Saved merged model to {MERGED_PATH}")


if __name__ == "__main__":
    main()
