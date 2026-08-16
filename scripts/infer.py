import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "pulsar-coder-1.5b" / "final"


def load(use_adapter: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    if use_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    return model


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a TypeScript function `debounce` that delays invoking a function until after a wait time."
    use_adapter = "--base" not in sys.argv

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load(use_adapter)
    messages = [
        {"role": "system", "content": "You are an expert software engineer. Given an instruction, write the requested code."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False, temperature=None, top_p=None)
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
