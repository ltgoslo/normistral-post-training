"""Downloads the LIMA dataset (Zhou et al., 2023) and writes its training split to
`lima_english.jsonl`. The dataset is gated on the Hugging Face Hub: accept its terms and set
the HF_TOKEN environment variable first."""
import os
import json

from datasets import load_dataset


def main():
    dataset = load_dataset("GAIR/lima", split="train", token=os.environ.get("HF_TOKEN"))

    with open("lima_english.jsonl", "w") as f:
        for example in dataset:
            row = {"conversations": example["conversations"], "source": example["source"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
