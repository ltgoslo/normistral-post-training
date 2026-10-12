"""Downloads the No Robots dataset (Rajani et al., 2023) and writes the shuffled train+test
splits to `no_robots_english.jsonl`, one conversation per line."""
import os
import json
import random

from datasets import load_dataset

random.seed(42)


def main():
    dataset = load_dataset("HuggingFaceH4/no_robots", token=os.environ.get("HF_TOKEN"))

    samples = [example for split in ["train", "test"] for example in dataset[split]]
    random.shuffle(samples)

    with open("no_robots_english.jsonl", "w") as f:
        for example in samples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
