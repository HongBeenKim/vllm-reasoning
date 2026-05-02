# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/TIGER-Lab/AceCodeRM-7B
# Copyright 2024 TIGER-Lab.
# Copyright 2023 The vLLM team.
"""Inference-only AceCodeRM (Qwen2-based reward model with a Linear(H, 1) head).

The reference implementation lives in TRL/AceCoder as a `Qwen2ForCausalLM`
subclass with a `ValueHead` that is a single `nn.Linear(hidden_size, 1)`
applied to the last non-pad token's hidden state. This differs from
`Qwen2ForRewardModel` (Qwen2.5-Math-RM style), whose head is
`Linear(H, H) -> ReLU -> Linear(H, num_labels)` — incompatible weights.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.activations import PoolerIdentity
from vllm.model_executor.layers.pooler.seqwise import pooler_for_classify
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .interfaces_base import default_pooling_type
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, maybe_prefix


class ValueHead(nn.Module):
    """AceCoder ValueHead: a single Linear(hidden_size, 1) with bias.

    The reference implementation also wraps a Dropout, but that is a no-op
    at inference time and would only complicate weight-key matching.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dtype: torch.dtype | None,
        prefix: str = "",
    ):
        super().__init__()
        self.summary = ReplicatedLinear(
            hidden_size,
            1,
            bias=True,
            params_dtype=head_dtype,
            return_bias=False,
            prefix=maybe_prefix(prefix, "summary"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(hidden_states)


@default_pooling_type(seq_pooling_type="LAST")
class Qwen2ForCausalRM(nn.Module, SupportsLoRA, SupportsPP):
    """AceCoder-style reward model: Qwen2 backbone + Linear(H, 1) value head.

    Output: one scalar reward per sequence, taken at the last token.
    """

    is_pooling_model = True
    pooler: Pooler

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        config.num_labels = 1

        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.head_dtype = vllm_config.model_config.head_dtype

        self.v_head = ValueHead(
            hidden_size=config.hidden_size,
            head_dtype=self.head_dtype,
            prefix=maybe_prefix(prefix, "v_head"),
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_classify(pooler_config, act_fn=PoolerIdentity())

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        hidden_states = hidden_states.to(self.head_dtype)
        logits = self.v_head(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)
