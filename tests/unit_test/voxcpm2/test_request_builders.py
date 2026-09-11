# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 preprocessing: cloning-mode routing and prefill layout."""

from __future__ import annotations

import torch

from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.models.voxcpm2.hf_config import VoxCPM2RuntimeConfig
from sglang_omni.models.voxcpm2.payload_types import VoxCPM2State
from sglang_omni.models.voxcpm2.request_builders import (
    VoxCPM2PreprocessingContext,
    build_prefill_inputs,
    build_voxcpm2_state,
)

_TOKEN_IDS = {
    C.AUDIO_START_TOKEN: 101,
    C.AUDIO_END_TOKEN: 102,
    C.AUDIO_PROMPT_START_TOKEN: 103,
    C.AUDIO_PROMPT_END_TOKEN: 104,
}


class _FakeTokenizer:
    """Character-per-token stand-in; ids are the character codes."""

    def __call__(self, text):
        return {"input_ids": [ord(ch) for ch in text]}

    def convert_tokens_to_ids(self, token):
        return _TOKEN_IDS[token]


class _FakeRequest:
    def __init__(self, inputs, params=None, metadata=None):
        self.inputs = inputs
        self.params = params or {}
        self.metadata = metadata or {}


class _FakePayload:
    def __init__(self, request):
        self.request = request
        self.request_id = "req"
        self.data = {}


def _context():
    return VoxCPM2PreprocessingContext(
        config=VoxCPM2RuntimeConfig(
            model_path="fake",
            audio_vae={"sample_rate": 16000, "out_sample_rate": 48000},
            patch_size=4,
            feat_dim=64,
        ),
        tokenizer=_FakeTokenizer(),
    )


def _state_for(references):
    payload = _FakePayload(_FakeRequest({"text": "hi", "references": references}))
    return build_voxcpm2_state(payload, _context())


def test_reference_without_transcript_is_a_timbre_prefix():
    state = _state_for([{"audio_path": "ref.wav"}])
    assert state.reference_audio == "ref.wav"
    assert state.prompt_audio == ""
    assert state.prompt_text == ""


def test_reference_with_transcript_becomes_continuation_audio():
    state = _state_for([{"audio_path": "ref.wav", "text": "hello"}])
    assert state.prompt_audio == "ref.wav"
    assert state.reference_audio == ""
    assert state.prompt_text == "hello"


def test_continuation_prefixes_the_transcript_to_the_target_text():
    state = _state_for([{"audio_path": "ref.wav", "text": "hello"}])
    assert state.text_token.tolist()[: len("hello")] == [ord(c) for c in "hello"]


def test_text_always_ends_with_the_audio_start_token():
    state = _state_for([])
    assert int(state.text_token[-1]) == _TOKEN_IDS[C.AUDIO_START_TOKEN]


def _prefill(state):
    return build_prefill_inputs(
        state, tokenizer=_FakeTokenizer(), patch_size=4, feat_dim=64
    )


def test_prefill_masks_cover_every_position_exactly_once():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    state.ref_latents = torch.zeros((5, 4, 64))
    state.prompt_latents = torch.zeros((2, 4, 64))
    prefill = _prefill(state)

    total = int(prefill.text_token.shape[0])
    assert int(prefill.audio_feat.shape[0]) == total
    assert torch.equal(prefill.text_mask + prefill.audio_mask, torch.ones(total))


def test_reference_prefix_leads_and_prompt_audio_trails():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    state.ref_latents = torch.zeros((5, 4, 64))
    state.prompt_latents = torch.zeros((2, 4, 64))
    prefill = _prefill(state)

    assert int(prefill.text_token[0]) == _TOKEN_IDS[C.AUDIO_PROMPT_START_TOKEN]
    # note (Xinhao Tan): index 6 is 1 start token + 5 reference patches + 1 end.
    assert int(prefill.text_token[6]) == _TOKEN_IDS[C.AUDIO_PROMPT_END_TOKEN]
    assert prefill.audio_mask[-2:].tolist() == [1, 1]
    assert prefill.text_mask[-2:].tolist() == [0, 0]


def test_zero_shot_prefill_is_text_only():
    state = VoxCPM2State(text_token=torch.tensor([1, 2, 3], dtype=torch.int32))
    prefill = _prefill(state)
    assert prefill.text_mask.tolist() == [1, 1, 1]
    assert prefill.audio_mask.tolist() == [0, 0, 0]
