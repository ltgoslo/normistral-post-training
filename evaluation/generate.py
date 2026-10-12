"""Samples responses from a model with Hugging Face transformers (single GPU).

Writes one JSON line per sampled response: {"id", "prompt", "reference", "generated_text", ...}.
The default sampling parameters are the ones used throughout the paper (temperature 0.5,
top-p 0.9, top-k 64). See generate_vllm.py for a faster vLLM backend.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_data import DATASETS, load_prompts


def model_stem(model_name: str) -> str:
    parts = model_name.rstrip("/").split("/")
    return "_".join(parts[-2:]) if parts[-1].startswith("checkpoint_") else parts[-1]


def default_output(args) -> str:
    suffix = f"_{args.prompt_id}" if args.dataset == "norquad" else ""
    return f"outputs/{args.dataset}{suffix}_{model_stem(args.model_name)}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses with a pre-trained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model ID on the Hugging Face Hub or a local checkpoint.")
    parser.add_argument("--dataset", type=str, default="norrewrite", choices=DATASETS)
    parser.add_argument("--prompt_id", type=int, default=0, help="NorQuAD prompt template (0-4).")
    parser.add_argument("--n_samples", type=int, default=1, help="Responses sampled per prompt (16 for the fluency scores).")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file; defaults to outputs/{dataset}_{model}.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true", help="Skip prompts that are already in the output file.")
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output(args)
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=getattr(torch, args.dtype),
        token=os.environ.get("HF_TOKEN")
    ).cuda()
    model.eval()

    rows = load_prompts(args.dataset, args.prompt_id)

    done = set()
    if args.resume and os.path.exists(args.output):
        done = {json.loads(line)["prompt"] for line in open(args.output) if line.strip()}

    with open(args.output, "a" if args.resume else "w") as f:
        for row in tqdm(rows):
            if row["prompt"] in done:
                continue

            encoding = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_return_sequences=args.n_samples
                )

            for i in range(args.n_samples):
                generated_text = tokenizer.decode(outputs[i, input_ids.shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({**row, "generated_text": generated_text}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
