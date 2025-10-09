# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Amirhossein Kazemnejad and/or its affiliates
# Copyright 2025 Milad Aghajohari and/or its affiliates
# Copyright 2025 Kamran Chitsaz and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DelethinkPPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import ensure_divisible, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss_with_trace_lengths
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics_with_distribution,
    compute_group_metrics,
    compute_has_eos,
    compute_per_scores_metrics,
    compute_throughout_metrics,
    compute_timing_and_throughput_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import Role, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.registry import register as register_trainer
from verl.trainer.treetune_ppo.ray_trainer import RayTreetunePPOTrainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.torch_functional import pad_sequence_to_length

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register_trainer("delethink_ppo")
class RayDelethinkPPOTrainer(RayTreetunePPOTrainer):
    def init_workers(self):
        """
        Customized init_workers for Delethink PPO.

        **NOTE**:
        Almost everything in this method is the same as original PPO,
        we only change the agent_loop_manager class.
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                global_config=self.config,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        #########################################################
        # start of the customized code
        #########################################################

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManagerWithCustomWorker

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManagerWithCustomWorker(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

        #########################################################
        # end of the customized code
        #########################################################

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        dynamic_sampling_state = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                #########################################################
                # batch components:
                #########################################################
                # - batch (torch tensors):
                #   - input_ids (batch_size, prompt_len)
                #   - attention_mask (batch_size, prompt_len)
                #   - prompts (batch_size, prompt_len)
                #   - position_ids (batch_size, prompt_len)
                # - non_tensor_batch (numpy arrays):
                #   - raw_prompt_ids (batch_size, [...prompt_ids as a python list...])
                #   - stuff for reward modeling such as ground truth, etc.

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # patch agent_name to delethink
                batch.non_tensor_batch["agent_name"] = np.array(["delethink_agent"] * len(batch), dtype=object)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # add orig_prompt_len to batch
                if "raw_prompt_ids" in batch.non_tensor_batch:
                    orig_prompt_len = np.array(list(map(len, batch.non_tensor_batch["raw_prompt_ids"])))
                else:
                    orig_prompt_len = batch.batch["attention_mask"].sum(-1).cpu().numpy()

                batch.non_tensor_batch["orig_prompt_len"] = orig_prompt_len

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        #########################################################
                        # gen_batch_output components:
                        #########################################################
                        # - batch (torch tensors):
                        #   - prompts (batch_size, prompt_len)
                        #   - responses (batch_size, response_len)
                        #   - response_mask (batch_size, response_len)
                        #   - input_ids (batch_size, seq_len: prompt_len + response_len)
                        #   - attention_mask (batch_size, seq_len)
                        #   - position_ids (batch_size, seq_len)
                        # - non_tensor_batch (numpy arrays):
                        #   - __num_turns__ (batch_size)
                        #   - trace_prompt_ids (bsz, [prompt_turn1 (np.array), prompt_turn2 (np.array), ...])
                        #   - trace_response_ids (bsz, [response_turn1 (np.array), response_turn2 (np.array), ...])

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # repeat to align with repeated responses in rollout
                    batch.non_tensor_batch["orig_prompt_len"] = orig_prompt_len
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.non_tensor_batch["traj_uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        assert not self.config.reward_model.launch_reward_fn_async, (
                            "Async reward is not supported for delethink"
                        )
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        score_metrics_keys = [
                            key for key in batch.non_tensor_batch.keys() if key.startswith("score_metrics/")
                        ]
                        score_metrics = None
                        for key in score_metrics_keys:
                            score_metrics = batch.non_tensor_batch[key]
                            metrics.update(
                                {
                                    f"{key}/mean": np.mean(score_metrics),
                                    f"{key}/std": np.std(score_metrics),
                                    f"{key}/max": np.max(score_metrics),
                                    f"{key}/min": np.min(score_metrics),
                                    f"{key}/dist": score_metrics,
                                }
                            )
                        del score_metrics_keys
                        del score_metrics

                    if self.dynamic_sampling_enabled:
                        keep_generating, dynamic_sampling_state, batch, ds_metrics = self._filter_groups(
                            batch, dynamic_sampling_state
                        )
                        if keep_generating:
                            continue

                        metrics.update(ds_metrics)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            raise NotImplementedError("KL penalty is not supported for delethink")

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = self._compute_advantages(batch)

                    # compute per-trajectory metrics
                    trace_lengths = []
                    for item in batch:
                        trace_lengths.append(sum(map(len, item.non_tensor_batch["trace_response_ids"])))
                    batch.batch["trace_lengths"] = torch.tensor(trace_lengths, device=batch.batch["input_ids"].device)
                    del trace_lengths  # just to not polute the variables in this big scope

                    metrics.update(self._compute_delethink_metrics(batch))
                    metrics.update(compute_group_metrics(batch))

                    if self.config.algorithm.get("filter_overlong_trajectories", False):
                        num_before_filter = len(batch)
                        batch = self._filter_overlong_trajectories(batch)
                        metrics.update(
                            {
                                "overlong_traj_deactivation/num_filtered": num_before_filter - len(batch),
                                "overlong_traj_deactivation/ratio": (num_before_filter - len(batch))
                                / num_before_filter,
                            }
                        )
                        del num_before_filter

                    # convert intermediate delethink turns into normal episodes
                    before_flatten_bsz = len(batch)
                    with marked_timer("flatten_intermediate_turns", timing_raw, color="purple"):
                        batch = self._flatten_intermediate_turns(batch)
                        metrics.update({"delethink/batch_size_after_flatten": len(batch)})
                        # after flattening, episodes appear in the order of the original batch
                        # promp_1, traj_1, state-action-1
                        # promp_1, traj_1, state-action-2
                        # promp_1, traj_1, ...
                        # promp_1, traj_2, state-action-1
                        # promp_1, traj_2, state-action-2
                        # ...
                        # promp_2, traj_1, state-action-1
                        # promp_2, traj_1, state-action-2
                        # ...
                        # promp_n, traj_n, state-action-n

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    rollout_dump_freq = self.config.trainer.get("rollout_dump_freq", 0)
                    should_save_rollout = (
                        rollout_data_dir is not None
                        and rollout_dump_freq > 0
                        and (self.global_steps % rollout_dump_freq == 0 or self.global_steps == 1)
                    )
                    if should_save_rollout:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                            self._dump_delethink_generations(batch=batch, dump_path=rollout_data_dir)

                    # since we flatten the intermediate turns, the resulting
                    # batch size may not be divisible by the dp size.
                    batch = self._ensure_divisible_by_dp_size_and_training_batch_size(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if self.config.algorithm.delethink.fixed_num_optim_steps > 0:
                        batch.meta_info["fixed_num_optim_steps"] = self.config.algorithm.delethink.fixed_num_optim_steps

                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    if loss_agg_mode == "seq-mean-token-norm-trace-length":
                        if self.config.algorithm.delethink.get("use_flat_batch_correction_ratio", False):
                            batch.batch["trace_lengths"] = batch.batch["trace_lengths"].float() * (
                                before_flatten_bsz / len(batch)
                            )
                            metrics.update({"delethink/flat_batch_correction_ratio": len(batch) / before_flatten_bsz})

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        train_inp_batch = batch.select(batch_keys=batch.batch.keys(), non_tensor_batch_keys=[])
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(train_inp_batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        entropy_agg = agg_loss_with_trace_lengths(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                            trace_lengths=batch.batch["trace_lengths"],
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_rollout_probs_diff_metrics

                            metrics.update(calculate_rollout_probs_diff_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            train_inp_batch = batch.select(batch_keys=batch.batch.keys(), non_tensor_batch_keys=[])
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(train_inp_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(train_inp_batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        raise NotImplementedError("Critic is not supported for delethink")

                    full_episodes_dump_freq = self.config.trainer.get("full_episodes_dump_freq", 0)
                    should_dump_full_episodes = (
                        rollout_data_dir is not None
                        and full_episodes_dump_freq > 0
                        and (self.global_steps % full_episodes_dump_freq == 0 or self.global_steps == 1)
                    )
                    if should_dump_full_episodes:
                        save_dir = os.path.join(rollout_data_dir, "full_episodes", f"iter_{self.global_steps:06d}")
                        os.makedirs(save_dir, exist_ok=True)
                        batch.meta_info["config"] = OmegaConf.to_container(self.config, resolve=True)
                        batch.save_to_disk(os.path.join(save_dir, "batch.pkl"))
                        batch.meta_info.pop("config", None)

                    if self.use_critic or self.config.trainer.critic_warmup > 0:
                        raise NotImplementedError("Critic is not supported for delethink")

                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        train_inp_batch = batch.select(batch_keys=batch.batch.keys(), non_tensor_batch_keys=[])
                        actor_output = self.actor_rollout_wg.update_actor(train_inp_batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics_with_distribution(
                        batch=batch, use_critic=self.use_critic, compute_score_metrics=False
                    )
                )
                metrics.update(compute_timing_and_throughput_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                timing_raw = {}  # clear timing

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _compute_advantages(self, batch: DataProto) -> DataProto:
        norm_adv_by_std_in_grpo = self.config.algorithm.get(
            "norm_adv_by_std_in_grpo", True
        )  # GRPO adv normalization factor

        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        return batch

    def _flatten_intermediate_turns(self, batch: DataProto) -> DataProto:
        """Convert batch with delethink turns into a list of state-action rows (normal episodes)"""

        max_prompt_len = batch.batch["prompts"].shape[-1]
        max_response_len = batch.batch["responses"].shape[-1]

        traj_uid_to_idx = {d.non_tensor_batch["traj_uid"]: i for i, d in enumerate(batch)}

        tensors = {
            "prompts": [],
            "responses": [],
            "input_ids": [],
            "attention_mask": [],
            "response_mask": [],
            "position_ids": [],
            "token_level_scores": [],
            "advantages": [],
            "returns": [],
            "trace_lengths": [],
            "rollout_log_probs": [],
        }

        non_tensors = defaultdict(list)

        tensor_keys_to_include = set(tensors.keys()) - {"rollout_log_probs"}
        non_tensors_keys_to_ignore = ["trace_prompt_ids", "trace_response_ids", "trace_log_probs"]
        pad_token_id = self.tokenizer.pad_token_id

        def add_item(
            prompt_ids: np.ndarray,
            response_ids: np.ndarray,
            log_probs: Optional[np.ndarray],
            datum: DataProto,
            pad: bool,
            turn_idx: int,
        ):
            non_tensors["turn_idx"].append(turn_idx)
            for k, v in item.non_tensor_batch.items():
                if k not in non_tensors_keys_to_ignore:
                    non_tensors[k].append(v)

            if log_probs is not None:
                # Pad log_probs to the max response length
                assert len(log_probs) == len(response_ids)
                log_probs = torch.tensor(log_probs)
                log_probs = pad_sequence_to_length(log_probs, max_response_len, pad_token_id=pad_token_id)
                tensors["rollout_log_probs"].append(log_probs)

            if not pad:
                for k in tensor_keys_to_include:
                    tensors[k].append(datum.batch[k])
                return

            prompt_ids = torch.tensor(prompt_ids)
            response_ids = torch.tensor(response_ids)

            prompt = pad_sequence_to_length(prompt_ids, max_prompt_len, pad_token_id=pad_token_id, left_pad=True)
            attention_mask = pad_sequence_to_length(
                torch.ones_like(prompt_ids), max_prompt_len, pad_token_id=0, left_pad=True
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            response = pad_sequence_to_length(response_ids, max_response_len, pad_token_id=pad_token_id)
            response_mask = pad_sequence_to_length(torch.ones_like(response_ids), max_response_len, pad_token_id=0)
            response_position_ids = position_ids[..., -1:] + torch.arange(1, response_ids.shape[-1] + 1)
            response_position_ids = pad_sequence_to_length(response_position_ids, max_response_len, pad_token_id=0)

            advantages = datum.batch["advantages"][0] * response_mask.float()
            token_scores = torch.zeros_like(
                advantages, dtype=torch.float32
            )  # This is an intermediate turn, so no per-token scores
            returns = datum.batch["returns"][0] * response_mask.float()

            tensors["prompts"].append(prompt)
            tensors["responses"].append(response)
            tensors["input_ids"].append(torch.cat([prompt, response], dim=-1))
            tensors["attention_mask"].append(torch.cat([attention_mask, response_mask], dim=-1))
            tensors["position_ids"].append(torch.cat([position_ids, response_position_ids], dim=-1))
            tensors["response_mask"].append(response_mask)
            tensors["advantages"].append(advantages)
            tensors["returns"].append(returns)
            tensors["token_level_scores"].append(token_scores)
            tensors["trace_lengths"].append(datum.batch["trace_lengths"])

        for item in batch:
            trace_prompt_ids: list[np.ndarray] = item.non_tensor_batch["trace_prompt_ids"]
            trace_response_ids: list[np.ndarray] = item.non_tensor_batch["trace_response_ids"]
            trace_log_probs: list[np.ndarray] = item.non_tensor_batch.get("trace_log_probs", None)
            assert len(trace_prompt_ids) == len(trace_response_ids)

            # Add the current item, which is the last turn
            add_item(
                prompt_ids=trace_prompt_ids[-1],
                response_ids=trace_response_ids[-1],
                log_probs=trace_log_probs[-1] if trace_log_probs is not None else None,
                datum=item,
                pad=False,  # no padding for the current turn as it's already padded
                turn_idx=len(trace_prompt_ids) - 1,
            )

            # Add the rest of the turns
            for i in range(len(trace_prompt_ids) - 1):
                add_item(
                    prompt_ids=trace_prompt_ids[i],
                    response_ids=trace_response_ids[i],
                    log_probs=trace_log_probs[i] if trace_log_probs is not None else None,
                    datum=item,
                    pad=True,
                    turn_idx=i,
                )

        # rollout_log_probs may be empty if calculate_log_probs is False
        if len(tensors["rollout_log_probs"]) == 0:
            tensors.pop("rollout_log_probs")

        for k, v in tensors.items():
            tensors[k] = torch.stack(v, dim=0)
        for k, v in non_tensors.items():
            non_tensors[k] = np.array(v, dtype=None if isinstance(v[0], np.ndarray | float | int) else object)

        batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=batch.meta_info)
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        indices = list(range(len(batch)))
        indices.sort(
            key=lambda idx: (
                traj_uid_to_idx[batch.non_tensor_batch["traj_uid"][idx]],
                batch.non_tensor_batch["turn_idx"][idx],
            )
        )
        batch.reorder(torch.tensor(indices))

        return batch

    def _ensure_divisible_by_dp_size_and_training_batch_size(self, batch: DataProto, mode: str = "delete") -> DataProto:
        """Ensure the batch is divisible by dp_size"""
        dp_size = self.actor_rollout_wg.world_size
        real_ppo_mini_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        )

        if self.config.algorithm.delethink.fixed_num_optim_steps > 0:
            divisor = np.lcm.reduce(np.array([dp_size, self.config.algorithm.delethink.fixed_num_optim_steps])).item()
        else:
            divisor = np.lcm.reduce(np.array([dp_size, real_ppo_mini_batch_size])).item()

        # check if the batch size is divisible by the dp size, if not, add some samples to make it divisible
        bs = len(batch)
        remainder = bs % divisor
        if remainder == 0:
            return batch

        if mode == "delete":
            logger.warning(
                f"Batch size {bs} is not divisible by dp_size {dp_size} or "
                f"ppo_mini_batch_size {real_ppo_mini_batch_size}, "
                f"deleting {remainder} samples to make it divisible"
            )
        elif mode == "copy":
            # add some samples to make it divisible
            logger.warning(
                f"Batch size {bs} is not divisible by dp_size {dp_size} or "
                f"ppo_mini_batch_size {real_ppo_mini_batch_size}, "
                f"copying {divisor - remainder} samples to make it divisible"
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        adjusted_batch = ensure_divisible(batch, size_divisor=divisor, mode=mode)

        return adjusted_batch

    def _filter_overlong_trajectories(self, batch: DataProto) -> DataProto:
        """Filter out trajectories without eos"""
        # has_eos is a 0/1 tensor of shape (batch_size,)
        has_eos: torch.Tensor = compute_has_eos(batch, self.tokenizer.eos_token_id, return_numpy=False)

        # Create indices of samples that contain EOS
        has_eos_indices = has_eos.nonzero(as_tuple=False).squeeze(1)
        batch = batch.select_idxs(has_eos_indices)

        return batch

    @torch.no_grad()
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # patch agent_name to delethink
            test_batch.non_tensor_batch["agent_name"] = np.array(["delethink_agent"] * len(test_batch), dtype=object)

            # repeat test batch
            if "val_sampling_params.n" in test_batch.non_tensor_batch:
                repeat_times = test_batch.non_tensor_batch["val_sampling_params.n"]
            elif "val_sampling_params.n" in test_batch.batch:
                repeat_times = test_batch.batch["val_sampling_params.n"]
            else:
                repeat_times = None

            if repeat_times is not None:
                test_batch = test_batch.sample_level_repeat(repeat_times=repeat_times)
            else:
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

            # add orig_prompt_len to batch
            if "raw_prompt_ids" in test_batch.non_tensor_batch:
                orig_prompt_len = np.array(list(map(len, test_batch.non_tensor_batch["raw_prompt_ids"])))
            else:
                orig_prompt_len = test_batch.batch["attention_mask"].sum(-1).cpu().numpy()

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.non_tensor_batch["orig_prompt_len"] = orig_prompt_len
            test_batch.meta_info["validate"] = True

            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            # Compute response lengths
            if "response_lengths" in test_batch.batch.keys():
                response_lengths = test_batch.batch["response_lengths"]
            else:
                response_lengths = test_batch.batch["response_mask"].sum(dim=-1)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["response_lengths"].extend(response_lengths.cpu().tolist())
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
                reward_extra_infos_dict["delethink_num_turns"].extend(
                    (test_batch.non_tensor_batch["__num_turns__"] - 1).tolist()
                )

            # compute trace length
            trace_lengths = []
            for item in test_batch:
                trace_lengths.append(sum(map(len, item.non_tensor_batch["trace_response_ids"])))
            reward_extra_infos_dict["trace_lengths"].extend(trace_lengths)

            # compute has_eos_column
            has_eos_column = compute_has_eos(test_batch, self.tokenizer.eos_token_id)
            reward_extra_infos_dict["has_eos"].extend(has_eos_column.tolist())

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        data_sources = np.concatenate(data_source_lst, axis=0)

        self._maybe_log_val_generations(
            data_sources=data_sources,
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            gts=sample_gts,
            **{k: v for k, v in reward_extra_infos_dict.items() if k != "reward"},
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_delethink_metrics(self, unflattened_batch: DataProto) -> dict[str, float]:
        """compute per-trajectory metrics
        Args:
            unflattened_batch: DataProto (train_batch_size * rollout_n, ...)
        Returns:
            dict[str, float]: per-trajectory metrics
        """
        batch = unflattened_batch

        scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
        num_turns = batch.non_tensor_batch["__num_turns__"] - 1
        lengths = batch.batch["trace_lengths"].cpu().numpy()
        has_eos = compute_has_eos(batch, self.tokenizer.eos_token_id)

        metrics = {
            # Scores
            "delethink/trace_scores/mean": np.mean(scores),
            "delethink/trace_scores/std": np.std(scores),
            "delethink/trace_scores/max": np.max(scores),
            "delethink/trace_scores/min": np.min(scores),
            "delethink/trace_scores/dist": scores,
            # Num turns
            "delethink/trace_num_turns/mean": np.mean(num_turns),
            "delethink/trace_num_turns/std": np.std(num_turns),
            "delethink/trace_num_turns/max": np.max(num_turns),
            "delethink/trace_num_turns/min": np.min(num_turns),
            "delethink/trace_num_turns/dist": num_turns,
            # Lengths
            "delethink/trace_lengths/mean": np.mean(lengths),
            "delethink/trace_lengths/std": np.std(lengths),
            "delethink/trace_lengths/max": np.max(lengths),
            "delethink/trace_lengths/min": np.min(lengths),
            "delethink/trace_lengths/dist": lengths,
            # Has EOS
            "delethink/trace_has_eos/mean": np.mean(has_eos),
            "delethink/trace_has_eos/std": np.std(has_eos),
        }

        metrics.update(
            compute_per_scores_metrics(
                scores,
                {
                    "trace_lengths": lengths,
                    "trace_num_turns": num_turns,
                    "trace_has_eos": has_eos,
                },
            )
        )

        return metrics

    def _dump_delethink_generations(self, batch: DataProto, dump_path: str):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path + "_delethink", exist_ok=True)
        filename = os.path.join(dump_path + "_delethink", f"{self.global_steps}.jsonl")

        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        traj_id_to_input = defaultdict(dict)
        traj_id_to_output = defaultdict(dict)
        traj_id_to_score = defaultdict(lambda: 0.0)
        traj_id_to_item = dict()
        for i, item in enumerate(batch):
            traj_id = item.non_tensor_batch["traj_uid"]
            turn_idx = item.non_tensor_batch["turn_idx"]
            traj_id_to_input[traj_id][turn_idx] = inputs[i]
            traj_id_to_output[traj_id][turn_idx] = outputs[i]
            traj_id_to_score[traj_id] += item.batch["token_level_scores"].sum(-1).item()
            if traj_id not in traj_id_to_item:
                traj_id_to_item[traj_id] = item

        n = len(traj_id_to_input)
        base_data = {
            "input": [],
            "output": [],
            "gts": [],
            "score": [],
            "num_delethink_turns": [],
            "step": [self.global_steps] * n,
            "trace_length": [],
            "advantage": [],
        }

        sep = "\n\n\n" + "#" * 100 + "\n" + "#" * 100 + "\n" + "#" * 100 + "\n\n"

        template = (
            ("#" * 30 + "\n# DELETHINK TURN {del_turn_idx:02d}\n" + "#" * 30 + "\n")
            + "# input:\n"
            + ("-" * 30 + "\n")
            + "{input}\n"
            + ("#" * 30 + "\n# DELETHINK TURN {del_turn_idx:02d}\n" + "#" * 30 + "\n")
            + "# output:\n"
            + ("-" * 30 + "\n")
            + "{output}\n"
        )

        traj_indices = list(traj_id_to_input.keys())
        traj_indices.sort(key=lambda x: traj_id_to_item[x].non_tensor_batch["uid"])

        for traj_id in traj_indices:
            item = traj_id_to_item[traj_id]

            traj_inputs = traj_id_to_input[traj_id]
            traj_outputs = traj_id_to_output[traj_id]

            input_str = traj_inputs[0]
            output_str = traj_outputs[0]

            inp_out_lst = [
                template.format(del_turn_idx=i + 1, input=traj_inputs[i], output=traj_outputs[i])
                for i in range(1, len(traj_inputs))
            ]
            inp_out_str = sep.join(inp_out_lst)
            if len(inp_out_str) > 0:
                output_str += sep + inp_out_str

            base_data["input"].append(input_str)
            base_data["output"].append(output_str)
            base_data["gts"].append(item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None))
            base_data["score"].append(traj_id_to_score[traj_id])
            base_data["num_delethink_turns"].append(item.non_tensor_batch["__num_turns__"].item() - 1)
            base_data["trace_length"].append(item.batch["trace_lengths"].item())
            base_data["advantage"].append(item.batch["advantages"][0].item())

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        # log to cloud
        log_kwargs = {
            **base_data,
            "idx": list(range(n)),
            "_prefix": "train/generations_delethink",
            "_columns": ["idx", "input", "output", "gts", "score", "num_delethink_turns", "trace_length", "advantage"],
        }
        self.validation_generations_logger.log_in_single_table(
            self.config.trainer.logger, log_kwargs, self.global_steps
        )

        print(f"Dumped generations to {filename}")
