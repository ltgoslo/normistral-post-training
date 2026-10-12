import os
import time
from statistics import mean

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed._tensor import Replicate, DTensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from transformers import AutoModelForCausalLM, AutoTokenizer

from worker import Worker, rgetattr
from stableadamw import StableAdamW

if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


class WorkerTrainer(Worker):
    """Updates the policy with the REINFORCE-style loss of the paper (Equation 4) regularized by a
    Rao-Blackwellized KL divergence to the reference policy (Equation 6)."""

    def __init__(self, config):
        super().__init__(config)
        self.tick = 0
        self.trainer_pipeline_group = config.trainer_pipeline_group
        self.entropy_coefficient = config.entropy_coefficient
        self.kl_coefficient = config.kl_coefficient
        self.target_entropy = config.target_entropy
        self.save_every = config.save_every

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=config.device,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.logger.info(f"Trainer model {config.model_name} loaded.")

        self.full_mesh = config.full_mesh
        self.submesh = config.train_submesh

        self.model = self.parallelize_model(self.model)
        self.logger.info(f"Trainer model {config.model_name} parallelized.")

        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=config.device,
            low_cpu_mem_usage=True
        )
        self.reference_model.eval()
        self.reference_model = self.parallelize_model(self.reference_model)
        self.logger.info(f"Reference model {config.ref_model_name} loaded and parallelized.")

        _pre_dp_module_transform(self.model)
        self.model = DDP(self.model, device_mesh=self.full_mesh["dp"])
        self.parameter_names = [n for n, _ in self.model.module.named_parameters()]

        self.optimizer = ZeroRedundancyOptimizer(
            params=self.model.parameters(),
            optimizer_class=StableAdamW,
            parameters_as_bucket_view=True,
            process_group=self.submesh.get_group("dp"),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            kahan_sum=True
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )

    def parallelize_model(self, model):
        n_layers = model.config.num_hidden_layers
        return parallelize_module(
            module=model,
            device_mesh=self.full_mesh["tp"],
            parallelize_plan={
                "model.embed_tokens": ColwiseParallel(output_layouts=Replicate()),
                **{f"model.layers.{i}.self_attn.q_proj": ColwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.self_attn.k_proj": ColwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.self_attn.v_proj": ColwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.self_attn.o_proj": RowwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.mlp.up_proj": ColwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.mlp.gate_proj": ColwiseParallel() for i in range(n_layers)},
                **{f"model.layers.{i}.mlp.down_proj": RowwiseParallel() for i in range(n_layers)},
            },
        )

    def step(self):
        if self.tick == 0:
            # the first generation round: receive the responses, nothing to train on yet
            self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for barrier")
            for _ in range(4):
                dist.barrier()
            self.tick += 1

            if self.worker_rank == 0:
                self.next_pipeline_conversations = []
                for pipeline_offset in range(self.n_pipelines_per_trainer):
                    self.next_pipeline_conversations += self.receive_conversations("generator", self.n_prompts_per_message, pipeline_offset=pipeline_offset)

            self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for barrier")
            for _ in range(3):
                dist.barrier()
            self.tick += 1

        # the responses from the previous round are scored by the judges in the meantime
        if self.worker_rank == 0:
            conversations = []
            for pipeline_offset in range(self.n_pipelines_per_trainer):
                pipeline_conversations = self.next_pipeline_conversations[pipeline_offset * self.n_prompts_per_message:(pipeline_offset + 1) * self.n_prompts_per_message]
                pipeline_conversations = self.receive_conversations_from_judges(pipeline_conversations, pipeline_offset=pipeline_offset)
                conversations.extend(pipeline_conversations)
        else:
            conversations = None

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-responses barrier")
        dist.barrier()

        if self.worker_rank == 0:
            self.next_pipeline_conversations = []
            for pipeline_offset in range(self.n_pipelines_per_trainer):
                self.next_pipeline_conversations += self.receive_conversations("generator", self.n_prompts_per_message, pipeline_offset=pipeline_offset)

        conversations = self.broadcast_conversations(conversations, self.my_pipeline_ranks["trainer"][0], self.trainer_pipeline_group)

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for train barrier")
        dist.barrier()
        self.tick += 1

        self.logger.info(f"Trainer {self.pipeline_index}:{self.worker_rank} started training.")
        self.train(conversations)

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-weights barrier")
        dist.barrier()

        for pipeline_offset in range(self.n_pipelines_per_trainer):
            self.send_model_weights(self.model, self.parameter_names, "generator", pipeline_offset=pipeline_offset)

        if (self.tick - 1) % self.save_every == 0:
            self.save()

        self.logger.info(f"{self.role}:{self.pipeline_index}:{self.worker_rank} tick {self.tick} - waiting for transfer-scores barrier")
        dist.barrier()

    def get_advantages(self, rewards):
        """Rewards normalized by the mean and standard deviation of their group (Equation 3)."""
        rewards = torch.tensor(rewards).to(self.device)
        return (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-3)

    def entropy_from_logits(self, logits: torch.Tensor):
        pd = F.softmax(logits, dim=-1)
        return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    def train(self, conversations):
        total_losses, policy_losses, kl_terms, entropies, grad_norms, all_rewards = [], [], [], [], [], []
        start_time = time.perf_counter()
        n_accumulation_steps = self.config.global_batch_size // (self.config.n_pipelines // self.n_pipelines_per_trainer)

        advantages = []
        for conversation in conversations:
            rewards = [float(m["reward"]) for m in conversation.metadata]
            all_rewards.extend(rewards)
            advantages.append(self.get_advantages(rewards))

        advantages = torch.stack(advantages)
        advantage_std = torch.std(advantages, dim=-1).mean()
        dist.all_reduce(advantage_std, dist.ReduceOp.AVG, group=self.config.worker_type_groups["trainer"])

        for local_step, conversation in enumerate(conversations):
            prompt = self.tokenizer.apply_chat_template(
                conversation.history,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.view(1, -1).to(self.device)

            # the loss of a prompt is normalized by the total length of its responses (Equation 4)
            length_normalization = sum(response.shape[0] for response in conversation.responses)

            for i, response in enumerate(conversation.responses):
                response = response.view(1, -1).to(self.device)
                input_ids = torch.cat((prompt_ids, response), dim=1)

                with torch.no_grad():
                    logits_ref = self.reference_model(input_ids=input_ids[:, :-1]).logits
                    logits_ref = logits_ref[0, -response.shape[1]:, :].float()
                    logprobs_ref_full = F.log_softmax(logits_ref, dim=-1)

                logits = self.model(input_ids=input_ids[:, :-1]).logits
                logits = logits[0, -response.shape[1]:, :].float()
                logprobs_new_full = F.log_softmax(logits, dim=-1)
                logprobs_new = logprobs_new_full.gather(1, response[0, :].unsqueeze(-1)).squeeze(-1)

                # REINFORCE with group-normalized advantages; the samples are (almost) on-policy,
                # so no importance weighting or clipping is used
                policy_loss = -(logprobs_new * advantages[local_step, i]).sum()

                # Rao-Blackwellized KL divergence between the policy and the reference model
                kl_per_token = (logprobs_new_full.exp() * (logprobs_new_full - logprobs_ref_full)).sum(-1)
                kl_term = kl_per_token.sum()

                # optional soft entropy target (disabled with --entropy_coefficient 0)
                entropy = self.entropy_from_logits(logits)
                with torch.no_grad():
                    entropy_weight = (entropy.mean() - self.target_entropy).exp() - 1
                entropy_term = entropy_weight * entropy.sum()

                loss = policy_loss + self.kl_coefficient * kl_term + self.entropy_coefficient * entropy_term
                loss = loss / length_normalization / n_accumulation_steps
                loss.backward()

                with torch.no_grad():
                    total_losses.append(loss.item())
                    policy_losses.append(policy_loss.item() / response.shape[1])
                    kl_terms.append(kl_per_token.mean().item())
                    entropies.append(entropy.mean().item())

                    if local_step == 0 and i == 0 and self.worker_rank == 0 and self.pipeline_index == 0:
                        self.logger.info("First 5 input ids: " + ", ".join(str(t) for t in prompt_ids[0, :5].tolist()))
                        self.logger.info("Last 5 input ids: " + ", ".join(str(t) for t in prompt_ids[0, -5:].tolist()))
                        self.logger.info("First 5 response ids: " + ", ".join(str(t) for t in response[0, :5].tolist()))
                        self.logger.info("Last 5 response ids: " + ", ".join(str(t) for t in response[0, -5:].tolist()))
                        self.logger.info("First 5 logprobs: " + ", ".join(str(t) for t in logprobs_new[:5].tolist()))
                        self.logger.info("Last 5 logprobs: " + ", ".join(str(t) for t in logprobs_new[-5:].tolist()))

            if (local_step + 1) % n_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1_000_000_000)  # only measures the norm
                grad_norms.append(grad_norm.item())

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                self.logger.info(f"Trainer {self.pipeline_index}:{self.worker_rank} step {local_step + 1}/{len(conversations)}: loss {loss.item():.4f}, grad norm {grad_norm:.4f}")

        with torch.no_grad():
            average_metrics = torch.tensor([
                sum(total_losses),
                mean(kl_terms),
                mean(policy_losses),
                mean(entropies),
                mean(all_rewards),
                mean([response.shape[0] for conversation in conversations for response in conversation.responses]),
            ]).to(self.device)
            dist.all_reduce(average_metrics, dist.ReduceOp.AVG, group=self.config.worker_type_groups["trainer"])
            average_loss, average_kl_term, average_policy_loss, average_entropy, average_reward, average_response_length = average_metrics.tolist()

            max_metrics = torch.tensor([
                max(m["generation_time"] for conversation in conversations for m in conversation.metadata),
                max(m["judge_time"] for conversation in conversations for m in conversation.metadata),
                time.perf_counter() - start_time,
                max(grad_norms) if grad_norms else 0.0,
            ]).to(self.device)
            dist.all_reduce(max_metrics, dist.ReduceOp.MAX, group=self.config.worker_type_groups["trainer"])
            max_generation_time, max_judge_time, max_train_time, max_grad_norm = max_metrics.tolist()

            if self.global_rank == 0:
                wandb.log({
                    "train/average_loss": average_loss,
                    "train/average_policy_loss": average_policy_loss,
                    "train/average_kl_term": average_kl_term,
                    "train/average_reward": average_reward,
                    "stats/average_entropy": average_entropy,
                    "stats/average_response_length": average_response_length,
                    "stats/average_advantage_std": advantage_std.item(),
                    "stats/max_grad_norm": max_grad_norm,
                    "stats/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "time/max_generation_time": max_generation_time,
                    "time/max_judge_time": max_judge_time,
                    "time/max_train_time": max_train_time,
                })

    def save(self):
        """Gathers the tensor-parallel shards and saves the policy in the Hugging Face format."""
        step = self.tick - 2
        output_dir = f"{self.config.output_dir}/checkpoint_{step}"
        is_saving_rank = self.worker_rank == 0 and self.pipeline_index == 0
        if is_saving_rank:
            os.makedirs(output_dir, exist_ok=True)

        state_dict = {}
        for n in self.parameter_names:
            p = rgetattr(self.model.module, n)
            if hasattr(p, "_st_info"):
                dtensor = DTensor.from_local(p, p._st_info.mesh, p._st_info.placements, run_check=False)
                state_dict[n] = dtensor.full_tensor().cpu()
            elif hasattr(p.data, "_local_tensor"):
                state_dict[n] = p.data.full_tensor().cpu()
            else:
                state_dict[n] = p.cpu()

        if is_saving_rank:
            torch.save(state_dict, f"{output_dir}/pytorch_model.bin")
            self.model.module.config.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.logger.info(f"Checkpoint saved to {output_dir}")
