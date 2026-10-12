"""Pairwise LLM-as-a-judge evaluation of generated responses (NorRewrite, NorSummarize or the
Mímir fluency prompts), following the NorEval protocol.

The judge (Llama 3.3 70B by default) sees an instruction, two responses and a human-written
reference, and picks A, B or tie. Every ordered pair of models is judged (both orders, to cancel
the position bias). With --baseline, only the pairs involving the baseline model are judged; this
is how the NLG scores of the paper are computed (win rate against Llama 3.1 8B). Only the first
sampled response of every prompt is used. Aggregate the results with winrates.py.
"""
import argparse
import json
import os

import vllm
from vllm import LLM, SamplingParams
from huggingface_hub import login

from eval_data import DATASETS, load_prompts, row_prompt, row_reference, row_generated_text


SYSTEM_PROMPT = "You are a helpful assistant that helps us rate a Norwegian AI model's responses to instructions."
INSTRUCTION_TEMPLATE = """Decide which response from the Norwegian AI system following the instruction is better, considering the following questions:
1. Most importantly, the AI systems should always respond in Norwegian. If a response is not in Norwegian, then you should consider it incorrect -- such a response should always be rated lower than any (even incorrect) response in Norwegian.
2. Does the response precisely follow the instruction? For example, a response that includes unrelated information or does not fulfill the task is not precisely following the instruction. Compare each response with the provided human response to decide if a response faithfully follows the instruction, especially when the instruction asks for expected word count or format.
3. Is the response helpful? For example, if the instruction asks for a recipe for healthy food, and the response is a useful recipe, then you can consider it helpful.
4. Is the language of the response natural? For example, AI responses are often verbose or repetitive, which is not natural. Compare with the provided human response to decide whether a response is natural.
5. Is the response factual/accurate? AI responses often make up new information. For example, if the response claims that Jens Stoltenberg is the current prime minister of Norway, then you should consider it inaccurate. Compare with the provided human response to verify whether a response is factual and accurate, especially with numbers.
6. Based on your aesthetics, which one do you prefer? For example, you might prefer one poem over another poem.

Select the response A or B that you prefer, or select tie if the two responses are similarly good or bad. Note that the responses can be truncated (don't consider that as a mistake).

Here are three examples:

# Example 1:

## Instruction:
Omformulér følgende spørsmål: "Hva er hovedstaden i Frankrike?"

## Response A:
Hovedstaden i Frankrike er Paris.

## Response B:
Kan du fortelle meg navnet på byen som fungerer som hovedstaden i Frankrike?

## Human Response:
Hva heter Frankrikes hovedstad?

In this example, B paraphrases the question as asked by the instruction. In contrast, A does not follow instruction as it answers the question instead. Human Response also paraphrases the question, just in a slightly different way. To sum up, B is the best response because it follows the instruction.

## Which is best, A, B, or tie?
B

# Example 2:

## Instruction:
Bytt ut det første verbet med et synonym:
Jeg elsker å surfe

## Response A:
Jeg hater å surfe

## Response B:
I like to surf

## Human Response:
Jeg liker å surfe

Response A tries to follow the instruction as it changes the first verb of the sentence, but it uses an antonym instead of a synonym. The response B might be correct, but it is written in English, not Norwegian, and non-Norwegian responses should always be rated as worse. Human Response changes the first verb, "elsker" (love), into its synonym, "liker" (like), as asked by the instruction. In conclusion, A is better than B because it is written in Norwegian.

## Which is best, A, B, or tie?
A

# Example 3:

## Instruction:
Bytt ut det første verbet med et synonym:
Jeg elsker å surfe

## Response A:
Jeg hater å surfe

## Response B:
Jeg liker ikke å surfe

## Human Response:
Jeg liker å surfe

In this example, neither output is correct and the responses are very similar. Human Response changes the first verb into its synonym, as asked by the instruction. To conclude, both A and B are equally incorrect, so the answer is tie.

## Which is best, A, B, or tie?
tie

Now here is the real task, first describe the qualities of each response and then end your message by writing "## Which is best, A, B, or tie?" and selecting among: A, B, or tie.

# Task:

## Instruction:
{instruction}

## Response A:
{output_1}

## Response B:
{output_2}

## Human Response:
{output_human}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise LLM-as-a-judge comparison of generated responses.")
    parser.add_argument("--dataset", type=str, default="norrewrite", choices=[d for d in DATASETS if d != "norquad"])
    parser.add_argument("--predictions", type=str, nargs="+", required=True, help="JSONL files produced by generate.py / generate_vllm.py; the model name is the file name without the dataset prefix.")
    parser.add_argument("--baseline", type=str, default=None, help="Prediction file of the baseline model; if given, only pairs with the baseline are judged.")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file; defaults to {dataset}_judge_results.json")
    parser.add_argument("--log_dir", type=str, default="evaluation_logs", help="Directory for the per-pair JSONL logs.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.dataset}_judge_results.json"
    return args


def model_name(path: str, dataset: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = f"{dataset}_"
    return name[len(prefix):] if name.startswith(prefix) else name


def load_predictions(path: str):
    """prompt -> first sampled response and (if stored) the gold reference"""
    predictions = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row_prompt(row)
        if prompt not in predictions:
            predictions[prompt] = {"generated_text": row_generated_text(row).strip(), "reference": row_reference(row)}
    return predictions


def main():
    args = parse_args()

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    predictions = {model_name(path, args.dataset): load_predictions(path) for path in args.predictions}
    if args.baseline is not None:
        baseline = model_name(args.baseline, args.dataset)
        if baseline not in predictions:
            predictions[baseline] = load_predictions(args.baseline)
        candidates = [name for name in predictions if name != baseline]
        pairs = [(baseline, c) for c in candidates] + [(c, baseline) for c in candidates]
    else:
        pairs = [(a, b) for a in predictions for b in predictions if a != b]

    # gold references: stored with the predictions, or looked up in the dataset
    references = {}
    for model_predictions in predictions.values():
        for prompt, prediction in model_predictions.items():
            if prediction["reference"]:
                references.setdefault(prompt, prediction["reference"])
    all_prompts = {prompt for model_predictions in predictions.values() for prompt in model_predictions}
    if any(prompt not in references for prompt in all_prompts):
        for row in load_prompts(args.dataset):
            if row["reference"]:
                references.setdefault(row["prompt"], row["reference"])
    missing = [prompt for prompt in all_prompts if prompt not in references]
    assert not missing, f"No reference response found for {len(missing)} prompts, e.g. {missing[0]!r}"

    judge = LLM(model=args.judge_model, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = judge.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama 3 end-of-turn token
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id and eot_id not in stop_token_ids:
        stop_token_ids.append(eot_id)
    sampling_params = SamplingParams(temperature=0.2, max_tokens=2048, stop_token_ids=stop_token_ids)

    os.makedirs(args.log_dir, exist_ok=True)
    results = {}
    for model_1, model_2 in pairs:
        prompts = [prompt for prompt in predictions[model_1] if prompt in predictions[model_2]]
        if len(prompts) != len(predictions[model_1]) or len(prompts) != len(predictions[model_2]):
            print(f"Warning: {model_1} and {model_2} share only {len(prompts)} prompts ({len(predictions[model_1])} vs. {len(predictions[model_2])})", flush=True)
        print(f"\nComparing {model_1} and {model_2} on {len(prompts)} prompts\n", flush=True)

        comparisons, inputs = [], []
        for prompt in prompts:
            comparison = {
                "instruction": prompt,
                "reference_output": references[prompt],
                "prediction_1": predictions[model_1][prompt]["generated_text"],
                "prediction_2": predictions[model_2][prompt]["generated_text"],
            }
            comparisons.append(comparison)
            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INSTRUCTION_TEMPLATE.format(
                    instruction=comparison["instruction"],
                    output_1=comparison["prediction_1"],
                    output_2=comparison["prediction_2"],
                    output_human=comparison["reference_output"]
                )}
            ]
            inputs.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

        judgements = judge.generate(inputs, sampling_params=sampling_params)
        for comparison, judgement in zip(comparisons, judgements):
            comparison["judgement"] = judgement.outputs[0].text.strip()

        with open(f"{args.log_dir}/{args.dataset}_{model_1}_vs_{model_2}.jsonl", "w", encoding="utf-8") as f:
            for comparison in comparisons:
                f.write(json.dumps(comparison, ensure_ascii=False) + "\n")

        results[f"{model_1}|{model_2}"] = comparisons

    output = {
        "evaluation_settings": {
            "dataset": args.dataset,
            "judge_model": args.judge_model,
            "system_prompt": SYSTEM_PROMPT,
            "instruction_template": INSTRUCTION_TEMPLATE,
            "sampling_params": str(sampling_params),
            "vllm_version": vllm.__version__
        },
        "results": results
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
