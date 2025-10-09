import math
from collections import defaultdict

import torch

from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import compute_has_eos
from verl.workers.reward_manager import register
from verl.workers.reward_manager.vineppo import VineppoRewardManager, maybe_recover_orig_prompt_and_response


@register("delethink")
class DelethinkRewardManager(VineppoRewardManager):
    pass


@register("tenary_delethink")
class TenaryDelethinkRewardManager(DelethinkRewardManager):
    def __init__(
        self,
        *args,
        success_score: float = 1.0,
        failure_score: float = -1.0,
        clip_score: float = 0.0,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_score = clip_score
        self.success_score = success_score
        self.failure_score = failure_score
        self.enabled = enabled

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        assert "orig_prompt_len" in data.non_tensor_batch

        has_eos_column = compute_has_eos(data, self.tokenizer.eos_token_id, return_numpy=False)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            valid_prompt_ids, valid_response_ids = maybe_recover_orig_prompt_and_response(
                data_item, valid_prompt_ids, valid_response_ids
            )

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            is_clipped = has_eos_column[i] == 0
            is_correct = int(score) == 1
            assert int(score) in [0, 1]

            tenary_reward = self.success_score if is_correct else self.failure_score
            tenary_reward = self.clip_score if is_clipped else tenary_reward

            reward_extra_info["score_metrics/tenary_reward"].append(float(tenary_reward))

            if self.enabled:
                reward_extra_info["score_metrics/is_correct"].append(float(is_correct))
                reward_extra_info["score_metrics/is_clipped"].append(float(is_clipped))
                reward = tenary_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("chunk_encourage_delethink")
class ChunkEncourageDelethinkRewardManager(DelethinkRewardManager):
    def __init__(
        self,
        *args,
        continuation_score: float = 1.0,
        encouragement_coeff: float = 0.02,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.continuation_score = continuation_score
        self.encouragement_coeff = encouragement_coeff
        self.enabled = enabled

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        assert "orig_prompt_len" in data.non_tensor_batch
        assert "__num_turns__" in data.non_tensor_batch

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            valid_prompt_ids, valid_response_ids = maybe_recover_orig_prompt_and_response(
                data_item, valid_prompt_ids, valid_response_ids
            )

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            assistant_turns = num_turns - 1
            chunk_encouragement_reward = (assistant_turns * self.continuation_score)

            reward_extra_info["score_metrics/chunk_encouragement_reward"].append(float(chunk_encouragement_reward))

            if self.enabled:
                reward_extra_info["score_metrics/score"].append(float(score))
                reward = score + self.encouragement_coeff * chunk_encouragement_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("cosine_delethink")
class CosineDelethinkRewardManager(DelethinkRewardManager):
    def __init__(
        self,
        *args,
        min_value_wrong: float,
        max_value_wrong: float,
        min_value_correct: float,
        max_value_correct: float,
        max_length: int,
        delethink_gen_length: int,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct
        self.max_length = max_length
        self.delethink_gen_length = delethink_gen_length
        self.enabled = enabled

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        assert "orig_prompt_len" in data.non_tensor_batch
        assert "__num_turns__" in data.non_tensor_batch

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            valid_prompt_ids, valid_response_ids = maybe_recover_orig_prompt_and_response(
                data_item, valid_prompt_ids, valid_response_ids
            )

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            assert int(score) in [0, 1]
            
            assistant_turns = num_turns - 1
            seq_length = valid_response_length + (assistant_turns - 1) * self.delethink_gen_length
            if score == 1:
                min_value = self.min_value_correct
                max_value = self.max_value_correct
            else:
                # Yes, they are swapped. This is required for the cosine formula below
                # to work with negative numbers.
                min_value = self.max_value_wrong
                max_value = self.min_value_wrong

            progress = seq_length / self.max_length
            cosine = math.cos(progress * math.pi)
            cosine_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            reward_extra_info["score_metrics/cosine_reward"].append(float(cosine_reward))

            if self.enabled:
                reward_extra_info["score_metrics/score"].append(float(score))
                reward = cosine_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
