import os
import re
import json
import time
import fcntl

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "You are a helpful assistant that helps to grade a Norwegian AI model's responses to user requests."

# the judge is asked to end its evaluation with a score in the format "X/10"
SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10\b")


def initialize_llm_with_retry(config, max_retries=3):
    """Initializes vLLM, falling back to the standard download method if hf_transfer fails."""
    for attempt in range(max_retries):
        try:
            if attempt == max_retries - 1:
                print("Falling back to standard download method...")
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            return LLM(**config)
        except RuntimeError as e:
            if "hf_transfer" not in str(e):
                raise
            print(f"Attempt {attempt + 1} failed with hf_transfer error")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to initialize LLM after {max_retries} attempts")


class WorkerRewardJudge:
    """LLM-as-a-judge reward model (d-RLAIF). It reads the sampled responses of its pipeline from a
    JSON file, grades every response on a 1-10 scale with the prompt template from Appendix E.2 of
    the paper, and writes the scores back to another JSON file."""

    def __init__(self, config):
        self.config = config
        self.judge_index = config.judge_index
        self.pipeline_index = config.pipeline_index
        self.worker_rank = config.worker_rank
        self.default_reward = config.default_reward

        with open(config.judge_template, "r") as f:
            self.instruction_template = f.read().strip()

        self.judge = initialize_llm_with_retry({
            "model": config.judge_model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": config.n_gpus_per_judge,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.95,
        })
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.2,
        )

        self.communication_folder = f"tmp/{config.slurm_job_id}"
        if config.worker_rank == 0 and config.pipeline_index == 0:
            os.makedirs(self.communication_folder, exist_ok=True)
        os.makedirs("logs/judgments", exist_ok=True)

        self.step_number = 0

    def run(self):
        while True:
            self.step()

    def step(self):
        conversations = self.receive_conversations_from_interface()
        scores = self.score_responses(conversations)
        self.send_conversations_to_interface(scores)
        self.step_number += 1

    def score_responses(self, conversations):
        tokenizer = self.judge.get_tokenizer()
        start_time = time.perf_counter()

        prompts = []
        for conversation in conversations:
            for response in conversation:
                conversation_dict = {
                    "conversation_history": response["messages"][:-1],
                    "gold_response": response["target"].strip(),
                    "ai_response": response["messages"][-1]["content"].strip()
                }
                user_prompt = self.instruction_template.replace("{{input}}", json.dumps(conversation_dict, ensure_ascii=False, indent=2).strip())
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                prompts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))

        judgments = self.judge.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        judgments = [judgment.outputs[0].text.strip() for judgment in judgments]
        scores = [{"reward": self.parse_judgment_score(judgment)} for judgment in judgments]

        with open(f"logs/judgments/judge_{self.judge_index}_pipeline_{self.pipeline_index}_step_{self.step_number}.jsonl", "w") as f:
            j = 0
            for conversation in conversations:
                for response in conversation:
                    f.write(json.dumps({
                        "conversation": response["messages"],
                        "judgment": judgments[j],
                        "score": scores[j]["reward"]
                    }, ensure_ascii=False) + "\n")
                    j += 1

        judge_time = time.perf_counter() - start_time
        for score in scores:
            score["judge_time"] = judge_time

        structured_scores, counter = [], 0
        for conversation in conversations:
            structured_scores.append(scores[counter:counter + len(conversation)])
            counter += len(conversation)

        return structured_scores

    def receive_conversations_from_interface(self):
        expected_filename = f"{self.communication_folder}/conversations_{self.judge_index}_{self.pipeline_index}.json"
        while True:
            if os.path.exists(expected_filename):
                time.sleep(0.5)

                with open(expected_filename, "r") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    conversations = json.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)

                os.unlink(expected_filename)
                return conversations

            time.sleep(0.25)

    def send_conversations_to_interface(self, scores):
        tmp_target_filename = f"{self.communication_folder}/tmp_scored_conversations_{self.judge_index}_{self.pipeline_index}.json"
        target_filename = f"{self.communication_folder}/scored_conversations_{self.judge_index}_{self.pipeline_index}.json"

        with open(tmp_target_filename, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(scores, f)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

        os.rename(tmp_target_filename, target_filename)

    def parse_judgment_score(self, judgment: str) -> float:
        """Returns the last "X/10" score of the judgment, clipped to [0, 10]; falls back to
        `--default_reward` when no score can be parsed (usually caused by a malformed response)."""
        matches = SCORE_PATTERN.findall(judgment)
        if not matches:
            return self.default_reward

        score = int(float(matches[-1]))
        return float(max(0, min(score, 10)))
