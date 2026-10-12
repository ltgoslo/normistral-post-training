"""Samples responses from a model with vLLM; same interface and output format as generate.py."""
import argparse
import json
import os

from vllm import LLM, SamplingParams

from eval_data import DATASETS, load_prompts
from generate import default_output


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses with a pre-trained model (vLLM).")
    parser.add_argument("--model_name", type=str, required=True, help="Model ID on the Hugging Face Hub or a local checkpoint.")
    parser.add_argument("--dataset", type=str, default="norrewrite", choices=DATASETS)
    parser.add_argument("--prompt_id", type=int, default=0, help="NorQuAD prompt template (0-4).")
    parser.add_argument("--n_samples", type=int, default=1, help="Responses sampled per prompt (16 for the fluency scores).")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file; defaults to outputs/{dataset}_{model}.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output(args)
    return args


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n_samples
    )

    rows = load_prompts(args.dataset, args.prompt_id)
    messages = [
        tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]

    outputs = model.generate(messages, sampling_params=sampling_params, use_tqdm=True)

    with open(args.output, "w") as f:
        for row, output_batch in zip(rows, outputs):
            for output in output_batch.outputs:
                f.write(json.dumps({**row, "generated_text": output.text.strip()}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
