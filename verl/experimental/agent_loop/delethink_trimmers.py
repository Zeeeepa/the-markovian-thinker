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

import numpy as np

_trimmer_registry: dict[str, dict] = {}


def register(name):
    """Decorator to register a trimmer class with a given name.

    Args:
        name: `(str)`
            The name of the trimmer.
    """

    def decorator(cls):
        if name in _trimmer_registry and _trimmer_registry[name] != cls:
            raise ValueError(f"Trimmer {name} has already been registered: {_trimmer_registry[name]} vs {cls}")
        _trimmer_registry[name] = cls
        return cls

    return decorator


def get_trimmer_cls(name):
    """Get the trimmer class with a given name.

    Args:
        name: `(str)`
            The name of the trimmer.

    Returns:
        `(type)`: The trimmer class.
    """
    if name not in _trimmer_registry:
        raise ValueError(f"Unknown trimmer: {name}")
    return _trimmer_registry[name]


class Trimmer:
    def __init__(self, keep_first: int, keep_last: int, tokenizer) -> None:
        """
        Implements the “keep head + keep tail” strategy used by ProgressiveEvaluator.

        Parameters
        ----------
        keep_first : int
            How many leading tokens of the current answer to retain.
        keep_last : int
            How many trailing tokens of the current answer to retain.
        """
        self.tokenizer = tokenizer
        if keep_first < 0 or keep_last < 0:
            raise ValueError("`keep_first` and `keep_last` must be non-negative")

        self.keep_first = keep_first
        self.keep_last = keep_last
        assert keep_last > 0, "`keep_last` must be positive"

    def __call__(
        self,
        *,
        prompt_toks: np.ndarray,
        answer_toks: np.ndarray,
        trim_count: int = 0,
        max_trim_count: int = None,
    ) -> np.ndarray:
        raise NotImplementedError


@register("progressive")
class ProgressiveTokenTrimmer(Trimmer):
    # ------------------------------------------------------------------ #
    #                        core trimming logic                          #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        *,
        prompt_toks: np.ndarray,
        answer_toks: np.ndarray,
        trim_count: int = 0,
        max_trim_count: int = None,
    ) -> np.ndarray:
        """
        Return a possibly trimmed copy of ``total_answer_toks`` as a NumPy array.

        Notes
        -----
        - Inputs must be 1-D ``np.ndarray`` of dtype compatible with token ids (e.g., int32/int64).
        - If ``keep_last == 0`` or ``keep_first == 0``, the corresponding segment is empty.
        """
        # desired sliding‑window size
        window = self.keep_first + self.keep_last

        # nothing to trim yet — return as-is (no copy to avoid extra allocs)
        if len(answer_toks) <= window:
            raise ValueError(f"Answer tokens must be longer than the window, but got {len(answer_toks)} and {window}")

        # ----------------- trim to head+tail -------------------------- #
        head = answer_toks[: self.keep_first]
        tail = answer_toks[-self.keep_last :]

        return np.concatenate([head, tail])