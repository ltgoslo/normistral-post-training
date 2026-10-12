"""Translates conversations from English to Norwegian with an LLM-based translation model (vLLM).

This produced the main translations of the paper (Unbabel/Tower-Plus-72B) and the Seed-X
translations of the ablation in Section 4.5. The input is a JSONL file with one conversation per
line, either in the No Robots format (a `messages` list of {"role", "content"} dicts) or in the
LIMA format (a `conversations` list of alternating user/assistant strings). The output contains
the same rows with a (partially) translated `messages` list.

Examples (run from this directory, 8 GPUs for the 72B model):
    python3 translate_llm.py --input no_robots_english.jsonl --output no_robots_nob_prompts.jsonl --roles user
    python3 translate_llm.py --input no_robots_nob_prompts.jsonl --output no_robots_nob_responses.jsonl --roles assistant
    python3 translate_llm.py --input lima_english.jsonl --output lima_nob_prompts.jsonl --roles user
    python3 translate_llm.py --input lima_nob_prompts.jsonl --output lima_nob_responses.jsonl --roles assistant
    python3 translate_llm.py --system seed-x --tensor_parallel_size 1 --input no_robots_english.jsonl --output no_robots_nob_seed-x.jsonl
"""
import argparse
import json
import warnings

from tqdm import tqdm
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore", category=UserWarning, module="vllm")

DEFAULT_MODELS = {
    "tower": "Unbabel/Tower-Plus-72B",
    "seed-x": "ByteDance-Seed/Seed-X-PPO-7B",
}
LANGUAGE_NAMES = {
    "nob": "Norwegian (Bokmål)",
    "nno": "Norwegian (Nynorsk)",
}
TOWER_TEMPLATE = "Translate the following English source text to {target_language}:\nEnglish: {source_text}\n{target_language}: "
SEED_X_TEMPLATE = "Translate the following text from English to Norwegian Bokmål:\n{source_text} <nb>"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file.")
    parser.add_argument("--system", type=str, default="tower", choices=list(DEFAULT_MODELS))
    parser.add_argument("--model", type=str, default=None, help="Model name; defaults to the standard model of --system.")
    parser.add_argument("--roles", type=str, default="all", choices=["user", "assistant", "all"], help="Which messages to translate.")
    parser.add_argument("--target_language", type=str, default="nob", choices=list(LANGUAGE_NAMES))
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of messages per vLLM call.")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.system]
    if args.system == "seed-x" and args.target_language != "nob":
        parser.error("The Seed-X prompt only supports Norwegian Bokmål.")
    return args


def to_messages(row):
    if "messages" in row:
        return row["messages"]
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(row["conversations"])
    ]


def translate_batch(source_texts, args, llm, sampling_params):
    if args.system == "tower":
        language = LANGUAGE_NAMES[args.target_language]
        prompts = [
            [{"role": "user", "content": TOWER_TEMPLATE.format(target_language=language, source_text=text)}]
            for text in source_texts
        ]
        outputs = llm.chat(prompts, sampling_params, use_tqdm=False)
    else:
        prompts = [SEED_X_TEMPLATE.format(source_text=text) for text in source_texts]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    return [output.outputs[0].text.strip() for output in outputs]


def main():
    args = parse_args()

    sampling_params = SamplingParams(temperature=0, max_tokens=8192, skip_special_tokens=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)

    rows = [json.loads(line) for line in open(args.input)]
    for row in rows:
        row["messages"] = to_messages(row)

    # (row index, message index, text) of every message that should be translated
    tasks = [
        (i, j, message["content"])
        for i, row in enumerate(rows)
        for j, message in enumerate(row["messages"])
        if args.roles == "all" or message["role"] == args.roles
    ]
    print(f"Translating {len(tasks)} messages from {len(rows)} conversations", flush=True)

    translations = []
    for start in tqdm(range(0, len(tasks), args.batch_size), desc="Translating"):
        batch = tasks[start:start + args.batch_size]
        translations += translate_batch([text for _, _, text in batch], args, llm, sampling_params)

    for (i, j, _), translation in zip(tasks, translations):
        rows[i]["messages"][j] = {"role": rows[i]["messages"][j]["role"], "content": translation}

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Translation complete, output written to {args.output}")


if __name__ == "__main__":
    main()
