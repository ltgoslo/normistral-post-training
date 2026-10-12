import json
import time
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from worker import Worker
from conversation_encoder import Conversation


class WorkerGenerator(Worker):
    """Samples responses from the (delayed) policy and sends them to the trainer and the judge."""

    def __init__(self, config):
        super().__init__(config)
        self.tick = 0

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"Tokenizer & model loaded: {config.model_name}")

        # every generator GPU works on its own disjoint slice of the prompts
        generator_index = self.pipeline_index * self.config.n_gpus_per_generator + self.worker_rank
        n_generators = self.n_pipelines * self.config.n_gpus_per_generator

        self.prompts, self.targets = [], []
        for i, line in enumerate(open(config.prompt_file, "r")):
            if i % n_generators != generator_index:
                continue
            conversation = json.loads(line)["messages"]

            prompt = conversation[:-1]  # the last (English) message is the gold response shown to the judge
            target = conversation[-1]["content"].strip()

            if sum(len(message["content"]) for message in prompt) > 8192:
                continue
            if any(message["role"] not in ["user", "assistant", "system"] for message in prompt):
                continue
            if prompt[-1]["role"] != "user":
                continue

            self.prompts.append(prompt)
            self.targets.append(target)

        self.random = random.Random(config.seed + generator_index)
        self.iteration_indices = []

    def step(self):
        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for generate barrier")
        dist.barrier()
        self.tick += 1

        conversations, targets, prompts = self.get_next_prompt_batch()
        token_ids, logprobs, metadata = self.generate_responses(prompts)
        conversations = [
            Conversation(
                history=conversation,
                responses=tokens,
                logprobs=token_logprobs,
                target=target,
                metadata=meta
            )
            for conversation, tokens, token_logprobs, target, meta in zip(conversations, token_ids, logprobs, targets, metadata)
        ]
        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-weights barrier")
        dist.barrier()

        if self.tick >= 3 and self.config.worker_rank == 0:
            self.receive_model_weights(self.model, "trainer")

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-scores barrier")
        dist.barrier()

        if self.tick >= 3:
            self.broadcast_model_weights(self.model, self.my_pipeline_ranks["generator"][0], self.config.generator_pipeline_group)

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-responses barrier")
        dist.barrier()

        if self.worker_rank > 0:
            self.send_conversations("generator", conversations, worker_offset=0)
        else:
            for i in range(1, self.config.n_gpus_per_generator):
                conversations += self.receive_conversations("generator", n_conversations=self.n_prompts_per_generator_gpu, worker_offset=i)
            conversations = conversations[:self.config.n_prompts_per_message]
            self.send_conversations("trainer", conversations)
            self.send_conversations("judge", conversations)

    def get_next_prompt(self):
        if len(self.iteration_indices) == 0:
            self.iteration_indices = list(range(len(self.prompts)))
            self.random.shuffle(self.iteration_indices)
        item_index = self.iteration_indices.pop(0)

        conversation_history = self.prompts[item_index]
        target = self.targets[item_index]

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            return self.get_next_prompt()

        prompt_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        if prompt_length > 4096:
            return self.get_next_prompt()

        return conversation_history, target, prompt

    def get_next_prompt_batch(self):
        all_conversations, all_prompts, all_targets = [], [], []
        for _ in range(self.n_prompts_per_generator_gpu):
            conversation_history, target, prompt = self.get_next_prompt()
            all_conversations.append(conversation_history)
            all_targets.append(target)
            all_prompts.append(prompt)

        return all_conversations, all_targets, all_prompts

    @torch.inference_mode()
    def generate_responses(self, prompts):
        all_token_ids, all_logprobs, all_metadata = [], [], []
        start_time = time.perf_counter()

        for i, prompt in enumerate(prompts):
            all_token_ids.append([])
            all_logprobs.append([])
            all_metadata.append([])

            self.logger.info(f"Generating response for prompt {i}...")

            input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                use_cache=True,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_return_sequences=self.n_responses_per_prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                return_dict_in_generate=True
            )
            token_ids = outputs.sequences[:, input_ids.shape[1]:].cpu()
            input_ids = input_ids.cpu()

            logprobs_list, token_ids_list, metadata_list = [], [], []
            for k in range(token_ids.shape[0]):
                # cut the response after the first end-of-sequence token
                eos_index = (token_ids[k] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if eos_index.numel() > 0:
                    eos_index = eos_index[0].item() + 1
                else:
                    eos_index = token_ids.shape[1]
                token_ids_list.append(token_ids[k][:eos_index])

                # log-probabilities of the sampled tokens under the sampling policy
                all_ids = torch.cat([input_ids, token_ids_list[-1].unsqueeze(0)], dim=-1)
                logits = self.model(input_ids=all_ids[:, :-1].to(self.device)).logits
                logits = logits[0, -token_ids_list[-1].shape[0]:, :].float()
                logprobs = F.log_softmax(logits, dim=-1).cpu()
                logprobs = logprobs.gather(1, token_ids_list[-1].unsqueeze(-1)).squeeze(-1)

                logprobs_list.append(logprobs)
                metadata_list.append({})

                if i == 0 and k == 0 and self.worker_rank == 0 and self.pipeline_index == 0:
                    self.logger.info("First 5 input ids: " + ", ".join(str(t) for t in input_ids[0, :5].tolist()))
                    self.logger.info("Last 5 input ids: " + ", ".join(str(t) for t in input_ids[0, -5:].tolist()))
                    self.logger.info("First 5 response ids: " + ", ".join(str(t) for t in token_ids_list[-1][:5].tolist()))
                    self.logger.info("Last 5 response ids: " + ", ".join(str(t) for t in token_ids_list[-1][-5:].tolist()))
                    self.logger.info("First 5 logprobs: " + ", ".join(str(t) for t in logprobs[:5].tolist()))
                    self.logger.info("Last 5 logprobs: " + ", ".join(str(t) for t in logprobs[-5:].tolist()))

            all_token_ids[i] += token_ids_list
            all_logprobs[i] += logprobs_list
            all_metadata[i] += metadata_list

        generation_time = time.perf_counter() - start_time
        for metadata in all_metadata:
            for m in metadata:
                m["generation_time"] = generation_time

        return all_token_ids, all_logprobs, all_metadata
