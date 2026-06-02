"""
Evaluate all JSONL result files in the results/ directory.
Handles: Gemma (string keys, no ```json wrapper), Qwen (think blocks, int keys),
         Qwen3.6 (think blocks in labels too).
"""

import ast
import json
import re
from pathlib import Path

RESULTS_DIR = Path("/data1/lokesh/bep/inference/results")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def normalize_dict(d: dict) -> dict:
    """Normalize keys to int (where possible) and values to lowercase."""
    out = {}
    for k, v in d.items():
        nk = int(k) if str(k).strip().lstrip("-").isdigit() else k
        out[nk] = str(v).strip().lower()
    return out


def parse_json_block(inner: str) -> dict | None:
    """Try json.loads, then ast fallback on the content inside {}."""
    inner = inner.strip()
    # Ensure it's wrapped in braces
    if not inner.startswith("{"):
        inner = "{" + inner + "}"
    try:
        return normalize_dict(json.loads(inner))
    except Exception:
        pass
    # ast fallback: quote unquoted string values
    try:
        fixed = re.sub(r"(?<=[{,]\s*)(\w+)\s*:", r'"\1":', inner)   # quote keys
        fixed = re.sub(r':\s*([a-zA-Z_]\w*)', r': "\1"', fixed)      # quote values
        return normalize_dict(ast.literal_eval(fixed))
    except Exception:
        pass
    # test.py style: split on commas then colons
    try:
        raw = inner.strip("{} \n")
        raw = re.sub(r"\s+", " ", raw).replace("'", "").replace('"', "")
        formatted = "{" + ", ".join(
            f'{k.strip()}: "{v.strip()}"'
            for part in raw.split(",")
            for k, v in [part.split(":")]
        ) + "}"
        return normalize_dict(ast.literal_eval(formatted))
    except Exception:
        return None


def parse_prediction(text: str) -> dict | None:
    text = strip_thinking(text)
    # Try ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        result = parse_json_block(m.group(1))
        if result is not None:
            return result
    # Try bare { ... } (first occurrence)
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        return parse_json_block(m.group(0))
    return None


def parse_label(text: str) -> dict | None:
    text = strip_thinking(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    return parse_json_block(m.group(1))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_file(path: Path) -> dict:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    missing = extra = correct = wrong = total = 0
    mal_correct = ben_correct = mal_wrong = ben_wrong = parse_errors = 0

    for r in records:
        preds = parse_prediction(r["predict"])
        labels = parse_label(r["label"])
        if preds is None or labels is None:
            parse_errors += 1
            continue

        for k in labels:
            if k not in preds:
                missing += 1
            elif preds[k] == labels[k]:
                correct += 1
                if preds[k] == "malicious":
                    mal_correct += 1
                elif preds[k] == "benign":
                    ben_correct += 1
            else:
                wrong += 1
                if preds[k] == "malicious":
                    mal_wrong += 1
                elif preds[k] == "benign":
                    ben_wrong += 1
        for k in preds:
            if k not in labels:
                extra += 1
        total += len(labels)

    t = total or 1
    accuracy  = correct / t * 100
    precision = mal_correct / (mal_correct + mal_wrong)  if (mal_correct + mal_wrong) > 0  else 0.0
    recall    = mal_correct / (mal_correct + ben_wrong)  if (mal_correct + ben_wrong) > 0  else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(
        total=total,
        correct=correct,
        wrong=wrong,
        missing=missing,
        extra=extra,
        parse_errors=parse_errors,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    rows = []
    for path in files:
        print(f"Evaluating {path.name} ...", flush=True)
        m = evaluate_file(path)
        rows.append((path.stem, m))

    # Print table
    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Model':<{col_w}} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Total':>7} {'ParseErr':>9}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, m in rows:
        print(
            f"{name:<{col_w}}"
            f"{m['accuracy']:>9.2f}%"
            f"{m['precision']:>10.3f}"
            f"{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}"
            f"{m['total']:>7}"
            f"{m['parse_errors']:>9}"
        )
    print(sep)


if __name__ == "__main__":
    main()
