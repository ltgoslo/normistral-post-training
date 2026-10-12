"""Sentence-level translation of conversations with classic machine-translation models.

Used for the translation-quality ablation in Section 4.5 (NLLB-200, MADLAD-400 and OPUS-MT).
Every message is split into paragraphs and sentences, which are translated one by one.
The input file is split evenly across the SLURM tasks of a job (one GPU per task): task `r`
appends to `{output_prefix}_{r}.jsonl` and resumes from that file if it already exists.
Concatenate the shards in rank order afterwards, e.g.
    cat no_robots_nob_nllb_{0..7}.jsonl > no_robots_nob_nllb.jsonl
"""
import argparse
import json
import os

import nltk
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download("punkt_tab", quiet=True)

SYSTEMS = {
    "nllb": "facebook/nllb-200-3.3B",
    "madlad": "google/madlad400-10b-mt",
    "opus": "Helsinki-NLP/opus-mt-en-gem",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", type=str, required=True, choices=list(SYSTEMS))
    parser.add_argument("--input", type=str, default="no_robots_english.jsonl")
    parser.add_argument("--output_prefix", type=str, default=None, help="Defaults to no_robots_nob_{system}.")
    args = parser.parse_args()

    if args.output_prefix is None:
        args.output_prefix = f"no_robots_nob_{args.system}"
    return args


class Translator:
    def __init__(self, system, device):
        self.system = system
        self.device = device
        model_name = SYSTEMS[system]

        if system == "nllb":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("nob_Latn")
        elif system == "madlad":
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        else:
            from transformers import pipeline
            self.pipe = pipeline("translation", model=model_name, device=device)

    @torch.no_grad()
    def translate_sentence(self, sentence):
        if self.system == "opus":
            return self.pipe(f">>nob<< {sentence[:192].strip()}")[0]["translation_text"]

        text = f"<2no> {sentence}" if self.system == "madlad" else sentence
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        extra = {"forced_bos_token_id": self.forced_bos_token_id} if self.system == "nllb" else {}
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + 50,
            num_beams=1,
            do_sample=False,
            **extra,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def translate(self, text):
        translated_paragraphs = []
        for paragraph in text.split("\n"):
            sentences = [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
            translated_paragraphs.append(" ".join(self.translate_sentence(s) for s in sentences))
        return "\n".join(translated_paragraphs)


def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    gpus_per_node = max(1, int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count())))
    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    translator = Translator(args.system, device)

    rows = [json.loads(line) for line in open(args.input)]
    rows = rows[len(rows) * rank // world_size : len(rows) * (rank + 1) // world_size]

    output_path = f"{args.output_prefix}_{rank}.jsonl"
    n_done = sum(1 for _ in open(output_path)) if os.path.exists(output_path) else 0

    with open(output_path, "a") as f:
        for row in tqdm(rows[n_done:], desc=f"Translating (rank {rank})"):
            for message in row["messages"]:
                message["content"] = translator.translate(message["content"].strip())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
