"""Merges the responses of several models into the input format of the annotation tool: one JSON
line per prompt with a `responses` dict (model name -> response).

Example (recreates outputs_for_annotation/annotation_set.jsonl):
    cd outputs_for_annotation
    python3 ../collect_for_annotation.py --inputs mistral-Nemo=mistral.jsonl translated-SFT=sft.jsonl on-policy-RL=rl.jsonl --output annotation_set.jsonl --shuffle
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Collect model responses for annotation.")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="name=file pairs; the files must contain the same prompts in the same order.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="NTNU")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the prompts (seeded).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = {}
    total_words = defaultdict(int)
    for name_and_file in args.inputs:
        name, filename = name_and_file.split("=", 1)
        for i, line in enumerate(open(filename, encoding="utf-8")):
            data = json.loads(line)
            if i not in output:
                output[i] = {
                    "sample_id": i,
                    "dataset": args.dataset_name,
                    "prompt": data["prompt"].strip(),
                    "responses": {}
                }
            assert output[i]["prompt"] == data["prompt"].strip(), f"Prompt mismatch at line {i} of {filename}"
            output[i]["responses"][name] = data["generated_text"].strip()
            total_words[name] += len(data["generated_text"].split())

    for name, count in total_words.items():
        print(f"{name}: {count / len(output):.1f} words per response")

    output = list(output.values())
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(output)

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in output:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
