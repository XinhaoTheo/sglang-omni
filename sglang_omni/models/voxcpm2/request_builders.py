# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 preprocessing: prompt text assembly and request parameter resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.hf_config import VoxCPM2RuntimeConfig
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference


@dataclass
class VoxCPM2PreprocessingContext:
    config: VoxCPM2RuntimeConfig
    tokenizer: Any


_CONTEXT: VoxCPM2PreprocessingContext | None = None


def set_voxcpm2_preprocessing_context(context: VoxCPM2PreprocessingContext) -> None:
    global _CONTEXT
    _CONTEXT = context


def _get_context() -> VoxCPM2PreprocessingContext:
    if _CONTEXT is None:
        raise RuntimeError("VoxCPM2 preprocessing context is not initialized")
    return _CONTEXT


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _reference_source(reference: dict[str, Any]) -> str | None:
    for key in ("audio_path", "path", "url"):
        value = reference.get(key)
        if value:
            return str(value)
    return audio_data_uri_from_reference(reference)


def build_voxcpm2_state(
    payload: StagePayload, context: VoxCPM2PreprocessingContext
) -> VoxCPM2State:
    """Build the VoxCPM2 state from an incoming request."""
    inputs = _dict(payload.request.inputs)
    params = _dict(payload.request.params)
    tts_params = _dict(_dict(payload.request.metadata).get("tts_params"))
    engine_params = _dict(_dict(params.get("stage_params")).get("tts_engine"))

    target_text = str(inputs.get("text") or "").strip()
    if not target_text:
        raise ValueError("VoxCPM2 requires nonempty input text")

    references = inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("VoxCPM2 accepts at most one reference audio")
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )
    source = _reference_source(reference)
    reference_text = str(
        reference.get("text") or tts_params.get("ref_text") or ""
    ).strip()

    # note (Xinhao Tan): upstream picks the cloning mode based on whether the
    # reference audio has a transcript. With one, the model continues from the
    # reference audio; without one, it only copies the voice.
    # So we check for a transcript instead of making the caller pick a mode.
    prompt_audio = source if (source and reference_text) else ""
    reference_audio = source if (source and not reference_text) else ""
    prompt_text = reference_text if prompt_audio else ""

    text = prompt_text + target_text if prompt_text else target_text
    tokenizer = context.tokenizer
    audio_start_id = int(tokenizer.convert_tokens_to_ids(C.AUDIO_START_TOKEN))
    text_ids = list(tokenizer(text)["input_ids"]) + [audio_start_id]

    config = context.config
    return VoxCPM2State(
        sample_rate=config.sample_rate,
        out_sample_rate=config.out_sample_rate,
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        reference_audio=reference_audio,
        text_token=torch.tensor(text_ids, dtype=torch.int32),
        target_text_length=len(tokenizer(target_text)["input_ids"]),
        patch_size=config.patch_size,
        feat_dim=config.feat_dim,
        inference_timesteps=int(
            _first(
                engine_params.get("inference_timesteps"),
                tts_params.get("inference_timesteps"),
                params.get("inference_timesteps"),
                default=C.DEFAULT_INFERENCE_TIMESTEPS,
            )
        ),
        cfg_value=float(
            _first(
                engine_params.get("cfg_value"),
                tts_params.get("cfg_value"),
                params.get("cfg_value"),
                default=C.DEFAULT_CFG_VALUE,
            )
        ),
        min_len=int(_first(engine_params.get("min_len"), default=C.DEFAULT_MIN_LEN)),
        max_len=int(
            _first(
                engine_params.get("max_len"),
                tts_params.get("max_len"),
                params.get("max_len"),
                default=C.DEFAULT_MAX_LEN,
            )
        ),
        seed=_first(tts_params.get("seed"), params.get("seed"), default=None),
        stream=bool(params.get("stream")),
    )


def preprocess_voxcpm2_payload(payload: StagePayload) -> StagePayload:
    """Preprocessing-stage entry point: validate the request and tokenize text."""
    state = build_voxcpm2_state(payload, _get_context())
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


__all__ = [
    "VoxCPM2PreprocessingContext",
    "build_voxcpm2_state",
    "preprocess_voxcpm2_payload",
    "set_voxcpm2_preprocessing_context",
]
