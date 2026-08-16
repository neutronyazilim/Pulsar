import ast
import json
import re
from pathlib import Path

REPOS = {
    "python": [
        "/home/tux/claws-mouse-linux",
    ],
    "typescript": [],
    "kotlin": [],
}

SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "dist", "build", "__pycache__", ".next"}
MIN_BODY_CHARS = 30
MAX_BODY_CHARS = 4000

SYSTEM_PROMPT = "You are an expert software engineer. Given an instruction, write the requested code."


def iter_files(root: Path, suffix: str):
    for p in root.rglob(f"*{suffix}"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name.endswith(".d.ts") or ".test." in p.name or ".spec." in p.name:
            continue
        yield p


def humanize(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    s = s.replace("_", " ")
    return s.strip().lower()


def extract_python(root: str):
    examples = []
    for path in iter_files(Path(root), ".py"):
        try:
            src = path.read_text(errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_") and node.name != "__init__":
                continue
            doc = ast.get_docstring(node)
            try:
                body_src = ast.get_source_segment(src, node)
            except Exception:
                continue
            if not body_src or not (MIN_BODY_CHARS <= len(body_src) <= MAX_BODY_CHARS):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if doc and len(doc.strip()) >= 10:
                instruction = f"Write a Python {kind} `{node.name}` that: {doc.strip().splitlines()[0]}"
            else:
                if isinstance(node, ast.ClassDef):
                    instruction = f"Write a Python class `{node.name}` ({humanize(node.name)})."
                else:
                    args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                    arg_str = f" taking parameters ({', '.join(args)})" if args else ""
                    instruction = f"Write a Python function `{node.name}`{arg_str} that implements {humanize(node.name)}."
            examples.append({
                "language": "python",
                "instruction": instruction,
                "response": body_src.strip(),
                "source": str(path),
            })
    return examples


TS_DECL_RE = re.compile(
    r"(?P<jsdoc>/\*\*(?:(?!\*/).)*?\*/\s*)?"
    r"(?P<export>export\s+)?(?:default\s+)?(?P<async>async\s+)?"
    r"(?:"
    r"function\s+(?P<fname>\w+)\s*\((?P<fargs>[^)]*)\)[^{;]*\{"
    r"|const\s+(?P<cname>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?\((?P<cargs>[^)]*)\)\s*(?::[^=]+)?=>\s*\{"
    r"|class\s+(?P<klass>\w+)[^{]*\{"
    r")",
    re.DOTALL,
)


def brace_match(src: str, start: int) -> int:
    depth = 0
    i = start
    n = len(src)
    while i < n:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def jsdoc_summary(comment: str):
    lines = [
        l.strip().lstrip("*").strip()
        for l in comment.splitlines()
        if l.strip().lstrip("*").strip() and not l.strip().lstrip("*").strip().startswith("@")
    ]
    return lines[0] if lines else None


def extract_ts(root: str):
    examples = []
    for suffix in (".ts", ".tsx"):
        for path in iter_files(Path(root), suffix):
            try:
                src = path.read_text(errors="ignore")
            except Exception:
                continue
            for m in TS_DECL_RE.finditer(src):
                name = m.group("fname") or m.group("cname") or m.group("klass")
                if not name:
                    continue
                brace_pos = m.end() - 1
                end = brace_match(src, brace_pos)
                if end == -1:
                    continue
                code_start = m.start("export") if m.group("export") else (
                    m.start("async") if m.group("async") else
                    (m.start("fname") - len("function ") if m.group("fname") else
                     m.start("cname") - len("const ") if m.group("cname") else
                     m.start("klass") - len("class "))
                )
                code_start = max(code_start, (m.end("jsdoc") if m.group("jsdoc") else m.start()))
                body = src[code_start:end].strip()
                if not (MIN_BODY_CHARS <= len(body) <= MAX_BODY_CHARS):
                    continue
                summary = jsdoc_summary(m.group("jsdoc")) if m.group("jsdoc") else None
                if summary:
                    instruction = f"Write a TypeScript function/class `{name}` that: {summary}"
                else:
                    args = m.group("fargs") or m.group("cargs") or ""
                    kind = "class" if m.group("klass") else "function"
                    arg_str = f" with parameters ({args.strip()})" if args.strip() else ""
                    instruction = f"Write a TypeScript {kind} `{name}`{arg_str} that implements {humanize(name)}."
                examples.append({
                    "language": "typescript",
                    "instruction": instruction,
                    "response": body,
                    "source": str(path),
                })
    return examples


KT_DECL_RE = re.compile(
    r"(?P<kdoc>/\*\*(?:(?!\*/).)*?\*/\s*)?"
    r"(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|private|internal|protected)\s+)?(?:suspend\s+)?(?:override\s+)?"
    r"(?:fun\s+(?P<fname>\w+)\s*\((?P<fargs>[^)]*)\)[^{=]*\{"
    r"|(?:data\s+|sealed\s+|abstract\s+)?class\s+(?P<klass>\w+)[^{]*\{"
    r")",
    re.DOTALL,
)


def extract_kotlin(root: str):
    examples = []
    for path in iter_files(Path(root), ".kt"):
        try:
            src = path.read_text(errors="ignore")
        except Exception:
            continue
        for m in KT_DECL_RE.finditer(src):
            name = m.group("fname") or m.group("klass")
            if not name:
                continue
            brace_pos = m.end() - 1
            end = brace_match(src, brace_pos)
            if end == -1:
                continue
            code_start = m.end("kdoc") if m.group("kdoc") else m.start()
            body = src[code_start:end].strip()
            if not (MIN_BODY_CHARS <= len(body) <= MAX_BODY_CHARS):
                continue
            summary = jsdoc_summary(m.group("kdoc")) if m.group("kdoc") else None
            if summary:
                instruction = f"Write a Kotlin function/class `{name}` that: {summary}"
            else:
                kind = "class" if m.group("klass") else "function"
                args = m.group("fargs") or ""
                arg_str = f" with parameters ({args.strip()})" if args.strip() else ""
                instruction = f"Write a Kotlin {kind} `{name}`{arg_str} that implements {humanize(name)}."
            examples.append({
                "language": "kotlin",
                "instruction": instruction,
                "response": body,
                "source": str(path),
            })
    return examples


def to_chat(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": f"```{example['language']}\n{example['response']}\n```"},
        ]
    }


def main():
    all_examples = []
    for root in REPOS["python"]:
        all_examples += extract_python(root)
    for root in REPOS["typescript"]:
        all_examples += extract_ts(root)
    for root in REPOS["kotlin"]:
        all_examples += extract_kotlin(root)

    seen = set()
    deduped = []
    for ex in all_examples:
        key = (ex["language"], ex["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)

    out_path = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"
    with out_path.open("w") as f:
        for ex in deduped:
            f.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")

    by_lang = {}
    for ex in deduped:
        by_lang[ex["language"]] = by_lang.get(ex["language"], 0) + 1
    print(f"Wrote {len(deduped)} examples to {out_path}")
    print("By language:", by_lang)


if __name__ == "__main__":
    main()
