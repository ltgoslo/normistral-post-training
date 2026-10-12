"""Judges the correctness of generated NorQuAD answers with an LLM (the NLU evaluation of the paper).

The judge compares every generated answer with the gold answer and outputs [[correct]] or
[[incorrect]]. The verdict is added to each row of the input files (keys "result" and
"judge_output"); compute the accuracy afterwards with norquad_accuracy.py.
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams
from huggingface_hub import login

from eval_data import row_reference, row_generated_text


SYSTEM_PROMPT = "You are a helpful assistant that helps us evalute a Norwegian AI model's responses to instructions."
INSTRUCTION_TEMPLATE = """Please evaluate whether the following response is accurate or not. The evaluated model was asked to perform reading comprehension and provide an answer based on the provided context. Below, you will find the question, the correct reference answer and the model's answer. Does the model's answer accurately address the question? First, compare the model's answer with the reference answer. Then, on a new line, provide your evaluation as either "[[correct]]" or "[[incorrect]]".

**Question:**
{question}

**Reference Answer:**
{output_human}

**Model's Answer:**
{output_model}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Judge the correctness of NorQuAD answers.")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="JSONL files produced by generate.py --dataset norquad (updated in place).")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    return parser.parse_args()


def parse_verdict(text: str):
    for line in reversed(text.split("\n")):
        if "[[correct]]" in line:
            return "correct"
        if "[[incorrect]]" in line:
            return "incorrect"
    return None


def main():
    args = parse_args()

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    judge = LLM(model=args.judge_model, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = judge.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama 3 end-of-turn token
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id not in stop_token_ids:
        stop_token_ids.append(eot_id)
    sampling_params = SamplingParams(temperature=0.2, max_tokens=2048, stop_token_ids=stop_token_ids)

    for path in args.inputs:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        print(f"Evaluating {path} with {len(rows)} samples", flush=True)

        prompts = []
        for row in rows:
            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTION_TEMPLATE.format(
                    question=row["question"].strip(),
                    output_human=row_reference(row),
                    output_model=row_generated_text(row).strip()
                )}
            ]
            prompts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

        judgements = judge.generate(prompts, sampling_params=sampling_params)
        for row, judgement in zip(rows, judgements):
            row["judge_output"] = judgement.outputs[0].text.strip()
            row["result"] = parse_verdict(row["judge_output"])

        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
