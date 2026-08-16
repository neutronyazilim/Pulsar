import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "data"
DATASET_PATH = ROOT / "data" / "dataset.jsonl"

SYSTEM_PROMPT = "You are an expert software engineer. Given an instruction, write the requested code."

LANG_MAP = {"python": "python", "typescript": "typescript", "kotlin": "kotlin"}
CAPS = {"python": 500, "typescript": 500, "kotlin": 350}
MIN_LEN, MAX_LEN = 80, 3000
MIN_SUBJECT_LEN = 15


def load_lang(lang: str, cap: int):
    path = RAW_DIR / lang / "data.jsonl"
    candidates = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            new_contents = d.get("new_contents", "")
            old_contents = d.get("old_contents", "")
            subject = d.get("subject", "").strip()
            if old_contents.strip():
                continue
            if not (MIN_LEN <= len(new_contents) <= MAX_LEN):
                continue
            if len(subject) < MIN_SUBJECT_LEN:
                continue
            candidates.append((subject, new_contents))
    random.shuffle(candidates)
    return candidates[:cap]


def to_chat(lang: str, subject: str, code: str):
    instruction = f"Write a {lang.capitalize()} file that: {subject}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"```{lang}\n{code.strip()}\n```"},
        ]
    }


def main():
    added = 0
    with DATASET_PATH.open("a") as out:
        for lang, cap in CAPS.items():
            samples = load_lang(lang, cap)
            for subject, code in samples:
                out.write(json.dumps(to_chat(lang, subject, code), ensure_ascii=False) + "\n")
                added += 1
            print(f"{lang}: added {len(samples)}")
    print(f"Total added: {added}")


if __name__ == "__main__":
    main()
