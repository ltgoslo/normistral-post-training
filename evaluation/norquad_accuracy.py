"""Prints the NorQuAD accuracy of files judged by judge_norquad.py, grouped by model.

Usage: python3 norquad_accuracy.py outputs/norquad_*.jsonl
"""
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

FILENAME_PATTERN = re.compile(r"^norquad_(\d+)_(.+)\.jsonl$")


def main(paths):
    results = defaultdict(dict)
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        judged = [row for row in rows if row.get("result") in ("correct", "incorrect")]
        if not judged:
            print(f"Skipping {path}: not judged yet")
            continue

        match = FILENAME_PATTERN.match(os.path.basename(path))
        model_name, prompt_id = (match.group(2), int(match.group(1))) if match else (os.path.basename(path), 0)
        accuracy = sum(row["result"] == "correct" for row in judged) / len(judged)
        results[model_name][prompt_id] = (accuracy, len(judged), len(rows) - len(judged))

    for model_name in sorted(results):
        print(f"\nModel: {model_name}")
        for prompt_id in sorted(results[model_name]):
            accuracy, n, n_unparsed = results[model_name][prompt_id]
            print(f"  prompt {prompt_id}: accuracy {accuracy:.2%} ({n} judged, {n_unparsed} unparsable judgements)")
        accuracies = [a for a, _, _ in results[model_name].values()]
        print(f"  mean accuracy over prompts: {mean(accuracies):.2%}, best prompt: {max(accuracies):.2%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
