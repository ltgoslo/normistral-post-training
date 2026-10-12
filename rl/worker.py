import os
import time
import json
import fcntl
import logging
import functools
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from conversation_encoder import ConversationEncoder, Conversation, ConversationTensor


def rgetattr(obj, attr):
    return functools.reduce(getattr, attr.split("."), obj)


class Worker:
    """Base class of the trainer, generator and idle workers.

    Trainers and generators of the same pipeline exchange conversations and model weights with
    torch.distributed point-to-point calls; the judges (separate vLLM processes outside of the
    process group) communicate through JSON files in `tmp/<SLURM job id>/`.
    """

    def __init__(self, config):
        self.config = config
        self.conversation_encoder = ConversationEncoder(config)

        self.role = config.role
        self.n_pipelines = config.n_pipelines
        self.pipeline_index = config.pipeline_index
        self.global_rank = config.global_rank
        self.worker_rank = config.worker_rank
        self.device = config.device
        self.n_prompts_per_message = config.n_prompts_per_message
        self.n_responses_per_prompt = config.n_responses_per_prompt
        self.n_prompts_per_generator_gpu = max(1, self.config.n_prompts_per_message // self.config.n_gpus_per_generator)
        self.n_pipelines_per_trainer = config.n_pipelines_per_trainer
        self.n_judges = config.n_judges
        if hasattr(config, "my_pipeline_ranks"):
            self.my_pipeline_ranks = config.my_pipeline_ranks
            self.worker_rank_ranges = config.worker_rank_ranges

        logging.basicConfig(
            format="%(asctime)s | [%(worker_role)s-%(pipeline_index)s-GPU%(gpu_index)s] | %(message)s",
            level=logging.INFO
        )
        self.logger = logging.LoggerAdapter(
            logging.getLogger("worker_comm"),
            {"worker_role": self.role, "pipeline_index": self.pipeline_index, "gpu_index": self.config.global_rank}
        )

        self.communication_folder = f"tmp/{config.slurm_job_id}"
        if config.worker_rank == 0 and config.pipeline_index == 0:
            os.makedirs(self.communication_folder, exist_ok=True)

        self.logger.info(f"Initialized {self.role} worker {self.pipeline_index} on GPU {self.config.global_rank}")

    def step(self):
        raise NotImplementedError("The step method should be implemented in subclasses.")

    def run(self):
        # give all processes (especially the vLLM judges) time to load their models before the
        # first synchronous step; the trainer starts last because it is waited for by the others
        time.sleep(10 * 60)
        if self.role != "trainer":
            time.sleep(2 * 60)
        self.logger.info(f"Starting {self.role} worker {self.pipeline_index} on GPU {self.config.global_rank}")
        while True:
            self.step()

    # --- tensor transfers ---

    def send_tensor(self, target_role: str, tensor: torch.Tensor, worker_offset=0, pipeline_offset=0):
        torch.cuda.synchronize()
        destination_global_rank = self.worker_rank_ranges[self.pipeline_index + pipeline_offset][target_role][worker_offset]
        dist.send(tensor.to(self.device), destination_global_rank)

    def receive_tensor(self, source_role: str, tensor_shape, tensor_dtype, worker_offset=0, pipeline_offset=0):
        source_global_rank = self.worker_rank_ranges[self.pipeline_index + pipeline_offset][source_role][worker_offset]
        tensor = torch.empty(tensor_shape, dtype=tensor_dtype, device=self.device)
        dist.recv(tensor, source_global_rank)
        torch.cuda.synchronize()
        return tensor

    def broadcast_tensor(self, tensor: torch.Tensor, source_rank: int, process_group: dist.ProcessGroup):
        torch.cuda.synchronize()
        tensor = tensor.to(self.device)
        dist.broadcast(tensor, src=source_rank, group=process_group)
        return tensor

    # --- conversation transfers ---

    def send_conversations(self, target_role: str, conversations: List[Conversation], worker_offset=0, pipeline_offset=0):
        self.logger.info(f"Sending conversations to {target_role}:{self.pipeline_index + pipeline_offset}:{worker_offset} from {self.role}:{self.pipeline_index}:{self.worker_rank}")

        if target_role == "judge":
            self.send_conversations_to_judges(conversations)
            return

        conversation_tensor = self.conversation_encoder.batch_encode(conversations)
        self.send_tensor(target_role, conversation_tensor.length_tensor, worker_offset, pipeline_offset)
        self.send_tensor(target_role, conversation_tensor.conversation_tensor, worker_offset, pipeline_offset)
        self.send_tensor(target_role, conversation_tensor.responses_tensor, worker_offset, pipeline_offset)
        self.send_tensor(target_role, conversation_tensor.logprobs_tensor, worker_offset, pipeline_offset)
        self.send_tensor(target_role, conversation_tensor.target_tensor, worker_offset, pipeline_offset)
        self.send_tensor(target_role, conversation_tensor.metadata_tensor, worker_offset, pipeline_offset)

    def receive_conversations(self, source_role: str, n_conversations: int, worker_offset=0, pipeline_offset=0):
        self.logger.info(f"Receiving conversations from {source_role}:{self.pipeline_index + pipeline_offset}:{worker_offset} to {self.role}:{self.pipeline_index}:{self.worker_rank}")

        lengths = self.receive_tensor(source_role, (n_conversations, 2 + self.n_responses_per_prompt), torch.long, worker_offset, pipeline_offset)
        conversation_tensor = self.receive_tensor(source_role, (n_conversations, 2, lengths[:, 0].max().item()), torch.long, worker_offset, pipeline_offset)
        responses_tensor = self.receive_tensor(source_role, (n_conversations, self.n_responses_per_prompt, lengths[:, 2:].max().item()), torch.long, worker_offset, pipeline_offset)
        logprobs_tensor = self.receive_tensor(source_role, (n_conversations, self.n_responses_per_prompt, lengths[:, 2:].max().item()), torch.float, worker_offset, pipeline_offset)
        target_tensor = self.receive_tensor(source_role, (n_conversations, max(1, lengths[:, 1].max().item())), torch.long, worker_offset, pipeline_offset)
        metadata_tensor = self.receive_tensor(source_role, (n_conversations, self.n_responses_per_prompt, self.conversation_encoder.n_metadata_keys), torch.float, worker_offset, pipeline_offset)

        conversation_tensor = ConversationTensor(
            length_tensor=lengths,
            conversation_tensor=conversation_tensor,
            responses_tensor=responses_tensor,
            logprobs_tensor=logprobs_tensor,
            target_tensor=target_tensor,
            metadata_tensor=metadata_tensor
        )
        return self.conversation_encoder.batch_decode(conversation_tensor)

    def broadcast_conversations(self, conversations: Optional[List[Conversation]], source_rank: int, process_group: dist.ProcessGroup):
        """Broadcasts the conversations held by `source_rank` to the other ranks of the process group
        (which pass `conversations=None`)."""
        self.logger.info(f"Broadcasting conversations from {self.role}:{self.pipeline_index}:{self.worker_rank}")
        n_conversations = self.n_prompts_per_message * self.n_pipelines_per_trainer

        if conversations is None:
            length_tensor = torch.empty((n_conversations, 2 + self.n_responses_per_prompt), dtype=torch.long, device=self.device)
            self.broadcast_tensor(length_tensor, source_rank, process_group)
            conversation_tensor = torch.empty((n_conversations, 2, length_tensor[:, 0].max().item()), dtype=torch.long, device=self.device)
            self.broadcast_tensor(conversation_tensor, source_rank, process_group)
            responses_tensor = torch.empty((n_conversations, self.n_responses_per_prompt, length_tensor[:, 2:].max().item()), dtype=torch.long, device=self.device)
            self.broadcast_tensor(responses_tensor, source_rank, process_group)
            logprobs_tensor = torch.empty((n_conversations, self.n_responses_per_prompt, length_tensor[:, 2:].max().item()), dtype=torch.float, device=self.device)
            self.broadcast_tensor(logprobs_tensor, source_rank, process_group)
            target_tensor = torch.empty((n_conversations, max(1, length_tensor[:, 1].max().item())), dtype=torch.long, device=self.device)
            self.broadcast_tensor(target_tensor, source_rank, process_group)
            metadata_tensor = torch.empty((n_conversations, self.n_responses_per_prompt, self.conversation_encoder.n_metadata_keys), dtype=torch.float, device=self.device)
            self.broadcast_tensor(metadata_tensor, source_rank, process_group)
            conversation_tensor = ConversationTensor(
                length_tensor=length_tensor,
                conversation_tensor=conversation_tensor,
                responses_tensor=responses_tensor,
                logprobs_tensor=logprobs_tensor,
                target_tensor=target_tensor,
                metadata_tensor=metadata_tensor
            )
            conversations = self.conversation_encoder.batch_decode(conversation_tensor)
        else:
            conversation_tensor = self.conversation_encoder.batch_encode(conversations)
            self.broadcast_tensor(conversation_tensor.length_tensor, source_rank, process_group)
            self.broadcast_tensor(conversation_tensor.conversation_tensor, source_rank, process_group)
            self.broadcast_tensor(conversation_tensor.responses_tensor, source_rank, process_group)
            self.broadcast_tensor(conversation_tensor.logprobs_tensor, source_rank, process_group)
            self.broadcast_tensor(conversation_tensor.target_tensor, source_rank, process_group)
            self.broadcast_tensor(conversation_tensor.metadata_tensor, source_rank, process_group)

        return conversations

    # --- weight transfers ---

    def send_model_weights(self, model, parameter_names, target_role: str, pipeline_offset=0):
        """Gathers the tensor-parallel shards and sends the full weights to the first worker of `target_role`."""
        destination_global_rank = self.worker_rank_ranges[self.pipeline_index + pipeline_offset][target_role][0]

        for n in parameter_names:
            p = rgetattr(model.module, n)

            if hasattr(p, "_st_info"):
                tensor = DTensor.from_local(p, p._st_info.mesh, p._st_info.placements, run_check=False).full_tensor()
            elif hasattr(p.data, "_local_tensor"):
                tensor = p.data.full_tensor()
            else:
                tensor = p

            if hasattr(tensor, "data"):
                tensor = tensor.data

            if self.worker_rank == 0:
                torch.cuda.synchronize()
                dist.send(tensor, destination_global_rank)

    @torch.no_grad()
    def receive_model_weights(self, model, source_role: str):
        if self.worker_rank != 0:
            return

        source_rank = self.my_pipeline_ranks[source_role][0]
        for param in model.parameters():
            in_tensor = torch.empty_like(param.data, device=self.device)
            dist.recv(in_tensor, source_rank)
            param.data.copy_(in_tensor)

    def broadcast_model_weights(self, model, source_rank: int, process_group: dist.ProcessGroup):
        for param in model.parameters():
            tensor = param.data.contiguous().to(self.device)
            dist.broadcast(tensor, src=source_rank, group=process_group)
            torch.cuda.synchronize()
            param.data.copy_(tensor)

    # --- file-based communication with the judges ---

    def send_conversations_to_judges(self, conversations: List[Conversation]):
        for i in range(self.n_judges):
            self.send_conversations_to_judge(conversations, i)

    def send_conversations_to_judge(self, conversations: List[Conversation], judge_index: int, pipeline_offset=0):
        conversations_list = []
        for conversation in conversations:
            responses_list = []
            for response in conversation.responses:
                response_text = self.conversation_encoder.decode_response(response)
                messages = conversation.history + [{"role": "assistant", "content": response_text.strip()}]
                responses_list.append({"messages": messages, "target": conversation.target})
            conversations_list.append(responses_list)

        tmp_target_filename = f"{self.communication_folder}/tmp_conversations_{judge_index}_{self.pipeline_index + pipeline_offset}.json"
        target_filename = f"{self.communication_folder}/conversations_{judge_index}_{self.pipeline_index + pipeline_offset}.json"

        with open(tmp_target_filename, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(conversations_list, f)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

        os.rename(tmp_target_filename, target_filename)  # atomic on most filesystems

    def receive_conversations_from_judges(self, conversations, pipeline_offset=0):
        """Waits for the scores of all judges and adds their average to the conversation metadata."""
        score_list = None
        for i in range(self.n_judges):
            new_score_list = self.receive_conversations_from_judge(i, pipeline_offset)
            if score_list is None:
                score_list = new_score_list
            else:
                for j in range(len(new_score_list)):
                    for k in range(len(new_score_list[j])):
                        for key, value in new_score_list[j][k].items():
                            score_list[j][k][key] += value

        for conversation, scores in zip(conversations, score_list):
            for metadata, score in zip(conversation.metadata, scores):
                metadata.update(score)

        return conversations

    def receive_conversations_from_judge(self, judge_index: int, pipeline_offset=0):
        expected_filename = f"{self.communication_folder}/scored_conversations_{judge_index}_{self.pipeline_index + pipeline_offset}.json"
        while True:
            if os.path.exists(expected_filename):
                time.sleep(0.5)

                with open(expected_filename, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    score_list = json.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)

                os.unlink(expected_filename)
                break

            time.sleep(0.25)

        for i, conversation in enumerate(score_list):
            for j, response in enumerate(conversation):
                for key, value in response.items():
                    score_list[i][j][key] = value / self.n_judges

        return score_list
