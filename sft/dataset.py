import json

import torch
from torch.utils.data import Dataset as TorchDataset


def to_messages(sample):
    """No Robots rows already contain `messages`; LIMA rows contain `conversations`,
    a list of alternating user/assistant strings."""
    if "messages" in sample:
        return sample["messages"]
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": message.strip()}
        for i, message in enumerate(sample["conversations"])
    ]


class Dataset(TorchDataset):
    """Tokenized conversations, sharded across the data-parallel ranks.
    The targets are -100 everywhere except on the assistant tokens."""

    def __init__(self, data_path: str, tokenizer, rank: int, world_size: int, seq_length: int = 4096):
        self.inputs, self.outputs = [], []

        for i, line in enumerate(open(data_path)):
            sample = json.loads(line)
            encoding = tokenizer.apply_chat_template(
                to_messages(sample),
                max_length=seq_length,
                truncation=True,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            assistant_mask = torch.as_tensor(encoding["assistant_masks"]).view_as(input_ids)
            output_ids = torch.where(assistant_mask == 1, input_ids, -100)

            # shift by one: predict token t+1 from tokens <= t
            input_ids, output_ids = input_ids[:, :-1], output_ids[:, 1:]

            if i == 0 and rank == 0:
                print("Example input:", tokenizer.decode(input_ids[0]))
                print("Example input tokens:", " ".join(tokenizer.convert_ids_to_tokens(input_ids[0])))
                print("Example target ids:", " ".join(str(t) for t in output_ids[0].tolist()), flush=True)

            if (output_ids == -100).all():  # the assistant turn was truncated away
                continue

            self.inputs.append(input_ids)
            self.outputs.append(output_ids)

        n = len(self.inputs) // world_size * world_size
        self.inputs = self.inputs[:n][rank::world_size]
        self.outputs = self.outputs[:n][rank::world_size]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
