# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 AR backbone on SGLang: the base and residual stacks in one model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.models.minicpm import MiniCPMDecoderLayer
from torch import nn

from sglang_omni.models.weight_loader import default_weight_loader


class _NoRope(nn.Module):
    """Stands in for a rotary embedding on the residual stack's layers."""

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        return q, k


def _stack_config(base: Any, *, num_layers: int) -> Any:
    """Copy an HF config for one stack, neutralizing SGLang's muP depth scaling."""
    config = base.__class__(**base.to_dict()) if hasattr(base, "to_dict") else base
    config.num_hidden_layers = num_layers
    # note (Xinhao Tan): do not restore the checkpoint's scale_depth here.
    # SGLang's MiniCPMDecoderLayer always multiplies each residual branch by
    # scale_depth / sqrt(num_hidden_layers), with no use_mup check, while
    # VoxCPM2 ships use_mup=False and adds the branch unscaled. Setting
    # scale_depth to sqrt(num_hidden_layers) makes that factor exactly 1.0.
    # Passing the real value silently scales every layer by ~0.265 instead.
    config.scale_depth = math.sqrt(num_layers)
    return config


class VoxCPM2SGLangModel(nn.Module):
    """The base and residual MiniCPM stacks sharing one paged KV pool.

    The two stacks advance in lockstep over identical positions and have
    identical KV geometry, so they are laid out as one flat list of layers with
    unique layer ids rather than two models with two caches.
    """

    _graph_feedback_buffer: torch.Tensor | None = None

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        lm_config = getattr(config, "lm_config", None)
        voxcpm_config = getattr(config, "voxcpm2_config", None)
        if lm_config is None or not isinstance(voxcpm_config, dict):
            raise ValueError("VoxCPM2 requires its lm_config and top-level config")

        self.num_base_layers = int(lm_config.num_hidden_layers)
        self.num_residual_layers = int(voxcpm_config.get("residual_lm_num_layers", 0))
        if self.num_residual_layers <= 0:
            raise ValueError("VoxCPM2 requires a positive residual_lm_num_layers")

        base_config = _stack_config(lm_config, num_layers=self.num_base_layers)
        residual_config = _stack_config(lm_config, num_layers=self.num_residual_layers)

        layers: list[nn.Module] = [
            MiniCPMDecoderLayer(
                base_config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.base_lm.layers.{layer_id}",
            )
            for layer_id in range(self.num_base_layers)
        ]
        for index in range(self.num_residual_layers):
            layer_id = self.num_base_layers + index
            layer = MiniCPMDecoderLayer(
                residual_config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.residual_lm.layers.{index}",
            )
            if voxcpm_config.get("residual_lm_no_rope", False):
                layer.self_attn.rotary_emb = _NoRope()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        hidden_size = int(lm_config.hidden_size)
        eps = float(lm_config.rms_norm_eps)
        from sglang.srt.layers.layernorm import RMSNorm

        self.base_norm = RMSNorm(hidden_size, eps=eps)
        self.residual_norm = RMSNorm(hidden_size, eps=eps)
        self._graph_feedback_buffer = None

    @property
    def graph_feedback_buffer(self) -> torch.Tensor | None:
        return self._graph_feedback_buffer

    def enable_graph_feedback(self, max_batch_size: int) -> None:
        """Own the static buffer the decode CUDA graph reads its input from.

        Every AR step's input embedding is produced by the local encoder from
        the previous step's sampled latent, so a captured graph must read from
        an address this model controls rather than from forward_batch.
        """
        if max_batch_size <= 0:
            raise ValueError("VoxCPM2 graph feedback buffer needs a positive size")
        parameter = next(self.parameters())
        self._graph_feedback_buffer = torch.zeros(
            (int(max_batch_size), int(self.config.lm_config.hidden_size)),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _run_stack(
        self,
        layers: Iterable[nn.Module],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        for layer in layers:
            hidden_states, _ = layer(positions, hidden_states, forward_batch, None)
        return hidden_states

    def forward_base(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        hidden_states = self._run_stack(
            self.layers[: self.num_base_layers], hidden_states, positions, forward_batch
        )
        return self.base_norm(hidden_states)

    def forward_residual(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        hidden_states = self._run_stack(
            self.layers[self.num_base_layers :], hidden_states, positions, forward_batch
        )
        return self.residual_norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            target = _map_checkpoint_name(name, self.num_base_layers)
            if target is None:
                continue
            parameter = params.get(target)
            if parameter is None:
                raise ValueError(
                    f"VoxCPM2 checkpoint weight {name!r} mapped to {target!r}, "
                    "which the AR model does not define"
                )
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(target)
        return loaded


def _map_checkpoint_name(name: str, num_base_layers: int) -> str | None:
    """Map a checkpoint parameter onto this model's flat layer list."""
    if name.startswith("base_lm.layers."):
        return f"layers.{name.removeprefix('base_lm.layers.')}"
    if name.startswith("residual_lm.layers."):
        rest = name.removeprefix("residual_lm.layers.")
        index, _, tail = rest.partition(".")
        return f"layers.{num_base_layers + int(index)}.{tail}"
    if name == "base_lm.norm.weight":
        return "base_norm.weight"
    if name == "residual_lm.norm.weight":
        return "residual_norm.weight"
    return None


EntryClass = VoxCPM2SGLangModel

__all__ = ["EntryClass", "VoxCPM2SGLangModel"]
