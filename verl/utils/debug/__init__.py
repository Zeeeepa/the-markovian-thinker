# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# APIs kept for backward compatibility purpose
# For new features please develop in verl/utils/profiler/
import torch
from tensordict import TensorDict

from verl import DataProto

from ..profiler import *  # noqa


def create_fake_gen_output(
    input_batch: DataProto, max_response_length: int, calculate_log_probs: bool = False
) -> DataProto:
    bsz = len(input_batch)
    device = input_batch.batch["input_ids"].device
    responses = torch.randint(10, 1000, (bsz, max_response_length), dtype=torch.long, device=device)
    response_mask = torch.ones(bsz, max_response_length, dtype=torch.long, device=device)
    input_ids = torch.cat([input_batch.batch["input_ids"], responses], dim=1)
    attention_mask = torch.cat([input_batch.batch["attention_mask"], response_mask], dim=1)
    response_position_ids = torch.arange(1, max_response_length + 1, device=device).unsqueeze(0).repeat(bsz, 1)
    response_position_ids = response_position_ids + input_batch.batch["position_ids"][:, -1:]
    position_ids = torch.cat([input_batch.batch["position_ids"], response_position_ids], dim=1)

    if calculate_log_probs:
        rollout_log_probs = torch.full((bsz, max_response_length), -0.99, dtype=torch.float32, device=device)
    else:
        rollout_log_probs = None

    batch = {
        "prompts": input_batch.batch["input_ids"],
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if calculate_log_probs:
        batch["rollout_log_probs"] = rollout_log_probs
    return DataProto(batch=TensorDict(batch, batch_size=bsz))
