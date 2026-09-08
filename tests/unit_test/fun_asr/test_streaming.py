# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.fun_asr.request_builders import (
    _retained_streaming_prefix,
    make_fun_asr_scheduler_adapters,
)
from sglang_omni.models.fun_asr.streaming import FunASRStreamingStrategy
from sglang_omni.proto import OmniRequest, StagePayload

_AUDIO_PAD = "<|object_ref_start|>"
_AUDIO_PAD_ID = 42


class _CharTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) for char in text])


@pytest.mark.parametrize(
    ("text", "rollback_chars", "expected_ids", "expected_text"),
    [
        ("", 8, [], ""),
        ("short", 8, [], ""),
        ("abcdef", 0, [97, 98, 99, 100, 101, 102], "abcdef"),
        ("abcdef", 2, [97, 98, 99, 100], "abcd"),
    ],
)
def test_retained_streaming_prefix_rolls_back_chars(
    text: str,
    rollback_chars: int,
    expected_ids: list[int],
    expected_text: str,
) -> None:
    assert _retained_streaming_prefix(_CharTokenizer(), text, rollback_chars) == (
        expected_ids,
        expected_text,
    )


class _BuilderTokenizer:
    eos_token_id = 151645
    vocab_size = 151936

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        if _AUDIO_PAD in text:
            audio_pad_count = text.count(_AUDIO_PAD)
            input_ids = (
                [10, 11, 12, 13, 14]
                + [_AUDIO_PAD_ID] * audio_pad_count
                + [15, 16, 17, 18]
            )
            return SimpleNamespace(input_ids=input_ids)
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == _AUDIO_PAD
        return _AUDIO_PAD_ID

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _feature_extractor(num_lfr_frames: int):
    def _call(
        audio,
        sampling_rate=None,
        return_tensors=None,
        return_attention_mask=True,
        padding="longest",
    ):
        return {
            "input_features": torch.zeros((1, 560, num_lfr_frames)),
            "attention_mask": torch.ones((1, num_lfr_frames), dtype=torch.long),
        }

    return _call


def test_request_builder_reconstructs_prefix_plus_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, result_adapter = make_fun_asr_scheduler_adapters(
        tokenizer=_BuilderTokenizer(),
        max_new_tokens=32,
        feature_extractor=_feature_extractor(17),
    )
    data = request_builder(
        StagePayload(
            request_id="fun-asr-streaming-refresh",
            request=OmniRequest(
                inputs={"audio_bytes": b"wav"},
                params={
                    "_asr_streaming": True,
                    "_asr_streaming_prefix_text": "abcdef",
                    "_asr_streaming_rollback_chars": 2,
                    "repetition_penalty": 1.3,
                },
            ),
            data={},
        )
    )
    data.output_ids = [101, 102]
    result = result_adapter(data)

    assert data.prompt_token_ids[-4:] == [97, 98, 99, 100]
    assert data.streaming_prefix_text == "abcd"
    assert data.req.sampling_params.repetition_penalty == 1.3
    assert result.data["text"] == "abcdef"


def test_request_builder_defaults_to_no_repetition_penalty_when_not_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, _ = make_fun_asr_scheduler_adapters(
        tokenizer=_BuilderTokenizer(),
        max_new_tokens=32,
        feature_extractor=_feature_extractor(17),
    )
    data = request_builder(
        StagePayload(
            request_id="fun-asr-offline",
            request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
            data={},
        )
    )

    assert data.streaming_prefix_text == ""
    assert data.req.sampling_params.repetition_penalty == 1.0


def test_fun_asr_strategy_waits_for_unfixed_chunks_before_using_prefix() -> None:
    strategy = FunASRStreamingStrategy()
    state = strategy.create_state(model_name="fun-asr-nano", language="English")

    first = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r0"
    )
    strategy.update_hypothesis(
        generated_text="hello wor", language="English", state=state
    )
    second = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r1"
    )
    strategy.update_hypothesis(
        generated_text="hello world", language="English", state=state
    )
    third = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r2"
    )

    assert first.extra_params["_asr_streaming_prefix_text"] is None
    assert first.sampling.repetition_penalty == 1.0
    assert second.extra_params["_asr_streaming_prefix_text"] is None
    assert third.extra_params["_asr_streaming_prefix_text"] == "hello world"
    assert third.extra_params["_asr_streaming_rollback_chars"] == 8
    assert third.sampling.repetition_penalty == 1.3


def test_fun_asr_strategy_updates_transcript_and_language() -> None:
    strategy = FunASRStreamingStrategy()
    state = strategy.create_state(model_name="fun-asr-nano", language=None)

    transcript = strategy.update_hypothesis(
        generated_text="hello world", language="English", state=state
    )

    assert transcript == "hello world"
    assert state.transcript == "hello world"
    assert state.language == "English"
    assert state.chunk_id == 1
