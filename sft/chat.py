#!/usr/bin/env python3
"""Minimal command-line chat with a model trained in this repository."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a language model from the command line")
    parser.add_argument("--model", type=str, required=True, help="Model ID on the Hugging Face Hub or a local path")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature; 0 means greedy decoding")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--single_turn", action="store_true", help="Forget the conversation history after every response")
    return parser.parse_args()


@torch.no_grad()
def generate_response(model, tokenizer, messages, max_new_tokens, temperature):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def main():
    args = parse_args()

    print(f"Loading model {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        print("Warning: this tokenizer has no chat template, the script may not work as expected.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device
    )

    print(f"\nYou are chatting with {args.model}")
    print("Type 'exit' or 'quit' to end the conversation, 'clear' to forget the history.\n", flush=True)

    messages = []
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if user_input.lower() == "clear":
                messages.clear()
                print("\nConversation history cleared.")
                continue
            if not user_input:
                continue

            if args.single_turn:
                messages.clear()
            messages.append({"role": "user", "content": user_input})

            print("\nAssistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages, args.max_new_tokens, args.temperature)
            print(response, flush=True)
            messages.append({"role": "assistant", "content": response})
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
