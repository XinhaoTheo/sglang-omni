# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 checkpoint configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.weight_loader import resolve_model_path


@dataclass
class VoxCPM2RuntimeConfig:
    """Normalized VoxCPM2 configuration for one checkpoint directory."""

    model_path: str
    lm: dict[str, Any] = field(default_factory=dict)
    encoder: dict[str, Any] = field(default_factory=dict)
    dit: dict[str, Any] = field(default_factory=dict)
    audio_vae: dict[str, Any] = field(default_factory=dict)
    patch_size: int = C.PATCH_SIZE
    feat_dim: int = C.FEAT_DIM
    residual_lm_num_layers: int = 0
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 0
    scalar_quantization_scale: int = 0
    max_length: int = 0
    dtype: str = "bfloat16"

    @property
    def sample_rate(self) -> int:
        return int(self.audio_vae.get("sample_rate", C.SAMPLE_RATE))

    @property
    def out_sample_rate(self) -> int:
        return int(self.audio_vae.get("out_sample_rate", C.OUT_SAMPLE_RATE))

    @property
    def latent_dim(self) -> int:
        return int(self.audio_vae.get("latent_dim", self.feat_dim))

    @property
    def cfm(self) -> dict[str, Any]:
        return dict(self.dit.get("cfm_config") or {})

    @property
    def dit_mean_mode(self) -> bool:
        return bool(self.dit.get("mean_mode", False))


def load_voxcpm2_config(
    model_path: str, *, local_files_only: bool = False
) -> VoxCPM2RuntimeConfig:
    """Read ``config.json`` from a VoxCPM2 checkpoint."""
    root = Path(resolve_model_path(model_path, local_files_only=local_files_only))
    with (root / C.CONFIG_FILE).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle) or {}

    def _section(key: str) -> dict[str, Any]:
        value = raw.get(key)
        return dict(value) if isinstance(value, dict) else {}

    return VoxCPM2RuntimeConfig(
        model_path=str(model_path),
        lm=_section("lm_config"),
        encoder=_section("encoder_config"),
        dit=_section("dit_config"),
        audio_vae=_section("audio_vae_config"),
        patch_size=int(raw.get("patch_size", C.PATCH_SIZE)),
        feat_dim=int(raw.get("feat_dim", C.FEAT_DIM)),
        residual_lm_num_layers=int(raw.get("residual_lm_num_layers", 0)),
        residual_lm_no_rope=bool(raw.get("residual_lm_no_rope", False)),
        scalar_quantization_latent_dim=int(
            raw.get("scalar_quantization_latent_dim", 0)
        ),
        scalar_quantization_scale=int(raw.get("scalar_quantization_scale", 0)),
        max_length=int(raw.get("max_length", 0)),
        dtype=str(raw.get("dtype", "bfloat16")),
    )


__all__ = ["VoxCPM2RuntimeConfig", "load_voxcpm2_config"]
