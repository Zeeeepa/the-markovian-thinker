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

import logging
import os
from dataclasses import field
from typing import Any
from uuid import uuid4

import numpy as np

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.delethink_trimmers import get_trimmer_cls
from verl.utils.profiler.performance import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_output_ids(response: dict[str, Any]) -> np.ndarray:
    if "output_ids" in response:
        return np.array(response["output_ids"])
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    _, output_token_ids = zip(*[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True)
    return np.array(output_token_ids)


class _DelethinkAgentLoopOutput(AgentLoopOutput):
    extra_fields: dict[str, Any] = field(default_factory=dict)


@register("delethink_agent")
class DelethinkAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level DelethinkAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns

        cls.keep_head = config.algorithm.delethink.keep_head
        cls.keep_tail = config.algorithm.delethink.keep_tail
        cls.intermediate_max_new_tokens = config.algorithm.delethink.intermediate_max_new_tokens

        cls.trimmer = None
        if config.algorithm.delethink.use_trimmer_class:
            cls.trimmer = get_trimmer_cls(config.algorithm.delethink.trimmer_name)(
                keep_first=cls.keep_head,
                keep_last=cls.keep_tail,
                tokenizer=tokenizer,
                **config.algorithm.delethink.trimmer_kwargs,
            )

        assert cls.intermediate_max_new_tokens <= cls.response_length, (
            f"intermediate_max_new_tokens must be less than or equal to response_length, "
            f"but got {cls.intermediate_max_new_tokens} and {cls.response_length}"
        )

        cls.calculate_log_probs = config.actor_rollout_ref.rollout.calculate_log_probs

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        orig_sampling_params = sampling_params

        prompt_ids = np.array(kwargs["raw_prompt_ids"])
        orig_prompt_ids = prompt_ids.copy()

        metrics = {}

        trace_prompt_ids = []
        trace_response_ids = []
        trace_log_probs = []

        assistant_turns = 0
        while True:
            sampling_params = orig_sampling_params.copy()
            sampling_params["return_full_output"] = True
            sampling_params["return_logprob"] = self.calculate_log_probs
            sampling_params["max_new_tokens"] = (
                self.intermediate_max_new_tokens if assistant_turns >= 1 else self.response_length
            )

            with simple_timer("generate_sequences", metrics):
                response: dict[str, Any] = await self.server_manager.generate(
                    request_id=uuid4().hex,  # we don't want to use the same server for the same trace
                    prompt_ids=prompt_ids.tolist(),  # sglang only accepts list[int]
                    sampling_params=sampling_params,
                )

            assistant_turns += 1

            response_ids = _get_output_ids(response)
            assert len(response_ids) <= sampling_params["max_new_tokens"], (
                f"Response ids must be less than or equal to max_new_tokens, "
                f"but got {len(response_ids)} and {sampling_params['max_new_tokens']} "
                f"for assistant turn {assistant_turns}"
            )

            trace_prompt_ids.append(prompt_ids)
            trace_response_ids.append(response_ids)
            if self.calculate_log_probs:
                log_probs = np.array([lp for lp, _, _ in response["meta_info"]["output_token_logprobs"]])
                assert len(log_probs) == len(response_ids)
                trace_log_probs.append(log_probs)

            finish_reason_is_eos = response["meta_info"]["finish_reason"]["type"] == "stop"
            if finish_reason_is_eos:
                break

            if assistant_turns >= self.max_assistant_turns:
                break

            if self.trimmer is None:
                prompt_ids = self._get_next_prompt_ids(prompt_ids, response_ids, assistant_turns)
            else:
                if assistant_turns == 1:
                    answer_toks = response_ids
                else:
                    answer_toks = np.concatenate([cut_response, response_ids])  # noqa: F821

                cut_response = self.trimmer(
                    prompt_toks=prompt_ids,
                    answer_toks=answer_toks,
                    trim_count=assistant_turns,
                    max_trim_count=self.max_assistant_turns
                    - 1,  # NOTE: number of trims is one less than max_assistant_turns
                )
                prompt_ids = np.concatenate([orig_prompt_ids, cut_response])

        assert len(response_ids) <= self.response_length, (
            f"Response ids must be less than or equal to response_length, "
            f"but got {len(response_ids)} and {self.response_length}"
        )

        assert len(trace_prompt_ids) == len(trace_response_ids) == assistant_turns, (
            f"Number of trace prompt ids and response ids must be equal to the number of assistant turns, "
            f"but got {len(trace_prompt_ids)} and {len(trace_response_ids)} and {assistant_turns}"
        )

        extra_fields = {
            "trace_prompt_ids": trace_prompt_ids,
            "trace_response_ids": trace_response_ids,
        }
        if self.calculate_log_probs:
            extra_fields["trace_log_probs"] = trace_log_probs

        output = _DelethinkAgentLoopOutput(
            prompt_ids=prompt_ids.tolist(),
            response_ids=response_ids.tolist(),
            response_mask=np.ones_like(response_ids).tolist(),
            num_turns=1 + assistant_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output

    def _get_next_prompt_ids(
        self, curr_prompt_ids: np.ndarray, curr_response_ids: np.ndarray, assistant_turn: int
    ) -> np.ndarray:
        if assistant_turn <= 1:
            # first turn
            # |--prompt--|
            # |--response--|
            # next_prompt
            # |--prompt--||h1||-tail-|

            tail = curr_response_ids[-self.keep_tail :]
            head = curr_response_ids[: self.keep_head]

            next_prompt_ids = np.concatenate([curr_prompt_ids, head, tail])
        else:
            # intermediate turns
            # |--prompt--||h1||-tail-|
            # |--resp--|
            # next_prompt
            # |--prompt--||h1||h2||-tail-|

            # Extract the tail from current prompt (everything after the last head)
            prompt_keep_len = len(curr_prompt_ids) - self.keep_tail + self.keep_head
            tail = curr_response_ids[-self.keep_tail :]

            # Construct next prompt: heads + new_head + new_tail
            next_prompt_ids = np.concatenate([curr_prompt_ids[:prompt_keep_len], tail])

        return next_prompt_ids
