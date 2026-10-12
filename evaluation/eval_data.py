"""Prompt sets and gold references shared by the evaluation scripts."""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

HF_DATASETS = {
    "norrewrite": "ltg/norrewrite-instruct",
    "norsummarize": "ltg/norsummarize-instruct",
}
PROMPT_FILES = {
    "fluency_prompts": "fluency_prompts.txt",  # the 100 Mímir prompts of the human evaluation and the fluency scores
    "extra_fluency_prompts": "extra_fluency_prompts.txt",  # 10 further Mímir prompts, used to train the annotators
}
DATASETS = list(HF_DATASETS) + list(PROMPT_FILES) + ["norquad"]

NORQUAD_PROMPTS = [
    "Tittel: {title}\n\nTekst:\n{passage}\n\nSpørsmål: {question}",
    'Tittel: {title}\n\nTekst:\n{passage}\n\nGitt teksten over, hva er svaret på følgende spørsmål? "{question}"',
    "Tittel: {title}\n\nTekst:\n{passage}\n\nSvar på følgende: {question}",
    'Tittel: {title}\n\nTekst:\n{passage}\n\nHvordan kan man svare på spørsmålet "{question}", gitt teksten over?',
    'Tittel: {title}\n\nTekst:\n{passage}\n\nGitt teksten over, besvar følgende spørsmål: "{question}"',
]


def load_mimir_references():
    """Instruction -> human-written output of the Mímir instruction dataset (de la Rosa et al., 2025)."""
    from datasets import load_dataset

    dataset = load_dataset("mimir-project/mimir-instruction", split="train")
    references = {}
    for example in dataset:
        if isinstance(example["Instruksjon"], str) and isinstance(example["Output"], str):
            references.setdefault(example["Instruksjon"].strip(), example["Output"].strip())
    return references


def load_prompts(dataset_name: str, prompt_id: int = 0):
    """Returns a list of {"id", "prompt", "reference"} dicts. NorQuAD rows additionally carry
    "title", "passage" and "question"; `prompt_id` selects one of its five prompt templates."""
    from datasets import load_dataset

    if dataset_name in HF_DATASETS:
        dataset = load_dataset(HF_DATASETS[dataset_name], split="test")
        return [
            {
                "id": example["id"],
                "prompt": f"{example['prompt'].strip()}\n\n{example['context'].strip()}",
                "reference": example["response"].strip(),
            }
            for example in dataset
        ]

    if dataset_name in PROMPT_FILES:
        references = load_mimir_references()
        with open(os.path.join(DATA_DIR, PROMPT_FILES[dataset_name])) as f:
            prompts = [line.strip() for line in f if line.strip()]
        return [{"id": str(i), "prompt": prompt, "reference": references.get(prompt)} for i, prompt in enumerate(prompts)]

    if dataset_name == "norquad":
        dataset = load_dataset("ltg/norquad", split="test")
        template = NORQUAD_PROMPTS[prompt_id]
        rows = []
        for example in dataset:
            title, _, passage = example["context"].partition("\n")
            title, passage = title.strip(), passage.strip()
            question = example["question"].strip()
            gold_answer = example["answers"]["text"][0].strip() if len(example["answers"]["text"]) > 0 else "N/A"
            rows.append({
                "id": example["id"],
                "prompt": template.format(title=title, passage=passage, question=question).strip(),
                "reference": gold_answer,
                "title": title,
                "passage": passage,
                "question": question,
            })
        return rows

    raise ValueError(f"Unknown dataset: {dataset_name}")


def row_prompt(row: dict) -> str:
    """The full user message of a generated row; also understands the raw NorRewrite/NorSummarize
    rows in outputs/final_comparison_*, which keep `prompt` and `context` separate."""
    if "context" in row:
        return f"{row['prompt'].strip()}\n\n{row['context'].strip()}"
    return row["prompt"].strip()


def row_reference(row: dict):
    """The gold response stored with a generated row, if any."""
    reference = row.get("reference", row.get("response", row.get("gold_answer")))
    return reference.strip() if isinstance(reference, str) else None


def row_generated_text(row: dict) -> str:
    return row["generated_text"] if "generated_text" in row else row["predicted_answer"]
