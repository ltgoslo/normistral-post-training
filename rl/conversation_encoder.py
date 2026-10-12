from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer


ID_TO_ROLE = ["user", "assistant", "system"]
ROLE_TO_ID = {role: i for i, role in enumerate(ID_TO_ROLE)}

ID_TO_METADATA = ["generation_time", "judge_time", "reward"]
METADATA_TO_ID = {metadata: i for i, metadata in enumerate(ID_TO_METADATA)}


class Conversation:
    """A prompt (`history`), the sampled responses with their sampling log-probabilities,
    the English gold response (`target`) and per-response metadata (timings, reward)."""

    def __init__(self, history: List[Dict[str, str]], responses: List[torch.Tensor], logprobs: List[torch.Tensor], target: Optional[str], metadata: List[Dict[str, float]]):
        self.history = history
        self.responses = responses
        self.logprobs = logprobs
        self.target = target
        self.metadata = metadata


class ConversationTensor:
    """Dense tensor encoding of a batch of conversations, used for torch.distributed transfers."""

    def __init__(self, length_tensor: torch.Tensor, conversation_tensor: torch.Tensor, responses_tensor: torch.Tensor, logprobs_tensor: torch.Tensor, target_tensor: torch.Tensor, metadata_tensor: torch.Tensor):
        self.length_tensor = length_tensor
        self.conversation_tensor = conversation_tensor
        self.responses_tensor = responses_tensor
        self.logprobs_tensor = logprobs_tensor
        self.target_tensor = target_tensor
        self.metadata_tensor = metadata_tensor


class ConversationEncoder:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.n_metadata_keys = len(ID_TO_METADATA)
        self.n_responses_per_prompt = config.n_responses_per_prompt

    def batch_encode(self, conversations: List[Conversation]) -> ConversationTensor:
        batch_size = len(conversations)

        individual_results = []
        max_conv_length, max_response_length, max_target_length = 0, 0, 0
        for conversation in conversations:
            conversation_tensor = self.conversation_to_tensor(conversation)
            individual_results.append(conversation_tensor)

            max_conv_length = max(max_conv_length, conversation_tensor.length_tensor[0].item())
            max_target_length = max(max_target_length, conversation_tensor.length_tensor[1].item())
            max_response_length = max(max_response_length, conversation_tensor.responses_tensor.shape[1])

        batch_length_tensor = torch.zeros((batch_size, 2 + self.n_responses_per_prompt), dtype=torch.long)
        batch_conversation_tensor = torch.zeros((batch_size, 2, max_conv_length), dtype=torch.long)
        batch_target_tensor = torch.zeros((batch_size, max(1, max_target_length)), dtype=torch.long)
        batch_responses_tensor = torch.zeros((batch_size, self.n_responses_per_prompt, max_response_length), dtype=torch.long)
        batch_logprobs_tensor = torch.zeros((batch_size, self.n_responses_per_prompt, max_response_length), dtype=torch.float)
        batch_metadata_tensor = torch.full((batch_size, self.n_responses_per_prompt, self.n_metadata_keys), float("nan"), dtype=torch.float)

        for i, conversation_tensor in enumerate(individual_results):
            batch_length_tensor[i] = conversation_tensor.length_tensor

            conv_length = conversation_tensor.length_tensor[0].item()
            batch_conversation_tensor[i, :, :conv_length] = conversation_tensor.conversation_tensor[:, :conv_length]

            target_length = conversation_tensor.length_tensor[1].item()
            if target_length > 0:
                batch_target_tensor[i, :target_length] = conversation_tensor.target_tensor[:target_length]

            response_length = conversation_tensor.responses_tensor.shape[1]
            batch_responses_tensor[i, :, :response_length] = conversation_tensor.responses_tensor
            batch_logprobs_tensor[i, :, :response_length] = conversation_tensor.logprobs_tensor

            batch_metadata_tensor[i, :, :] = conversation_tensor.metadata_tensor

        return ConversationTensor(
            length_tensor=batch_length_tensor,
            conversation_tensor=batch_conversation_tensor,
            responses_tensor=batch_responses_tensor,
            logprobs_tensor=batch_logprobs_tensor,
            target_tensor=batch_target_tensor,
            metadata_tensor=batch_metadata_tensor
        )

    def batch_decode(self, conversation_tensor: ConversationTensor) -> List[Conversation]:
        conversations = []
        for i in range(conversation_tensor.length_tensor.shape[0]):
            single_conversation_tensor = ConversationTensor(
                length_tensor=conversation_tensor.length_tensor[i],
                conversation_tensor=conversation_tensor.conversation_tensor[i],
                responses_tensor=conversation_tensor.responses_tensor[i],
                logprobs_tensor=conversation_tensor.logprobs_tensor[i],
                target_tensor=conversation_tensor.target_tensor[i],
                metadata_tensor=conversation_tensor.metadata_tensor[i]
            )
            conversations.append(self.tensor_to_conversation(single_conversation_tensor))
        return conversations

    def conversation_to_tensor(self, conversation: Conversation) -> ConversationTensor:
        content_tensors, role_tensors = [], []
        for message in conversation.history:
            content_tensor = self.tokenizer.encode(message["content"], add_special_tokens=False, return_tensors="pt").flatten()
            role_tensor = torch.full_like(content_tensor, ROLE_TO_ID[message["role"]], dtype=torch.long)
            content_tensors.append(content_tensor)
            role_tensors.append(role_tensor)

        conversation_tensor = torch.stack((torch.cat(content_tensors, dim=0), torch.cat(role_tensors, dim=0)), dim=0)

        if conversation.target is not None:
            target_tensor = self.tokenizer.encode(conversation.target, add_special_tokens=False, return_tensors="pt").flatten()
            target_length = target_tensor.shape[0]
        else:
            target_tensor = torch.tensor([], dtype=torch.long)
            target_length = 0

        response_lengths = [response.shape[0] for response in conversation.responses]
        max_response_length = max(response_lengths)
        responses_tensor = torch.zeros((self.n_responses_per_prompt, max_response_length), dtype=torch.long)
        logprobs_tensor = torch.zeros((self.n_responses_per_prompt, max_response_length), dtype=torch.float)
        for i, (response, logprobs) in enumerate(zip(conversation.responses, conversation.logprobs)):
            responses_tensor[i, :response.shape[0]] = response
            logprobs_tensor[i, :logprobs.shape[0]] = logprobs

        length_tensor = torch.tensor([conversation_tensor.shape[1], target_length] + response_lengths, dtype=torch.long)

        metadata_tensor = torch.full((self.n_responses_per_prompt, self.n_metadata_keys), float("nan"), dtype=torch.float)
        for i, metadata in enumerate(conversation.metadata):
            for key, value in metadata.items():
                metadata_tensor[i, METADATA_TO_ID[key]] = value

        return ConversationTensor(
            length_tensor=length_tensor,
            conversation_tensor=conversation_tensor,
            responses_tensor=responses_tensor,
            logprobs_tensor=logprobs_tensor,
            target_tensor=target_tensor,
            metadata_tensor=metadata_tensor
        )

    def tensor_to_conversation(self, tensor_encoding: ConversationTensor) -> Conversation:
        lengths = tensor_encoding.length_tensor.tolist()
        content_tensor = tensor_encoding.conversation_tensor[0, :lengths[0]]
        role_tensor = tensor_encoding.conversation_tensor[1, :lengths[0]]

        # segment the token sequence into messages wherever the role changes
        messages = []
        current_role, current_content_indices = None, []
        for i in range(len(role_tensor)):
            role_id = role_tensor[i].item()
            if role_id != current_role:
                if current_role is not None and current_content_indices:
                    content = self.tokenizer.decode(content_tensor[current_content_indices]).strip()
                    messages.append({"role": ID_TO_ROLE[current_role], "content": content})
                current_role = role_id
                current_content_indices = [i]
            else:
                current_content_indices.append(i)
        if current_role is not None and current_content_indices:
            content = self.tokenizer.decode(content_tensor[current_content_indices]).strip()
            messages.append({"role": ID_TO_ROLE[current_role], "content": content})

        if lengths[1] > 0:
            target = self.tokenizer.decode(tensor_encoding.target_tensor[:lengths[1]]).strip()
        else:
            target = None

        responses, logprobs = [], []
        for i in range(self.n_responses_per_prompt):
            responses.append(tensor_encoding.responses_tensor[i, :lengths[i + 2]])
            logprobs.append(tensor_encoding.logprobs_tensor[i, :lengths[i + 2]])

        metadatas = []
        for n in range(self.n_responses_per_prompt):
            metadata = {}
            for i, value in enumerate(tensor_encoding.metadata_tensor[n]):
                if not torch.isnan(value):
                    metadata[ID_TO_METADATA[i]] = value.item()
            metadatas.append(metadata)

        return Conversation(
            history=messages,
            responses=responses,
            logprobs=logprobs,
            target=target,
            metadata=metadatas
        )

    def decode_response(self, response_tensor: torch.Tensor) -> str:
        return self.tokenizer.decode(response_tensor, skip_special_tokens=True).strip()
