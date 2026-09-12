# SPDX-License-Identifier: Apache-2.0
"""Opt-in checkpoint parity for the VoxCPM2 ports against OpenBMB/VoxCPM.

The AR stacks run inside SGLang and are covered end to end; what this pins is
the hand-ported half - AudioVAE, local encoder, local DiT and the flow-matching
sampler - where a transcription slip produces plausible noise rather than an
error. Run it with the upstream package checked out:

    VOXCPM2_UPSTREAM_SOURCE=/path/to/VoxCPM \\
    VOXCPM2_PARITY_CHECKPOINT=openbmb/VoxCPM2 \\
    pytest tests/test_model/test_voxcpm2_parity.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.accelerator

# note (Xinhao Tan): tolerances are per dtype because bf16 keeps 8 mantissa
# bits - one ULP at magnitude 8 is already 0.03, so an fp32-sized tolerance
# reports rounding as a failed port. Seeds are fixed for the same reason a
# tolerance is: a run has to be comparable to the previous one.
_TOLERANCE = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (6e-2, 6e-2),
    torch.float16: (1e-2, 1e-2),
}
_SEED = 1234


def _seeded(*shape: int, device: str = "cuda:0", dtype=torch.float32):
    torch.manual_seed(_SEED)
    return torch.randn(*shape, device=device, dtype=dtype)


def _assert_close(ours: torch.Tensor, theirs: torch.Tensor, what: str) -> None:
    """Compare and report the deltas, so a passing run still yields numbers.

    The mismatch count separates a rounding difference from a wrong port: a
    handful of outliers is precision, most of the tensor is a real divergence.
    """
    atol, rtol = _TOLERANCE.get(theirs.dtype, (1e-4, 1e-4))
    dtypes = f"{ours.dtype}/{theirs.dtype}"
    ours = ours.detach().float().cpu()
    theirs = theirs.detach().float().cpu()
    assert ours.shape == theirs.shape, f"{what}: {ours.shape} vs {theirs.shape}"

    difference = (ours - theirs).abs()
    mismatched = int((difference > atol + rtol * theirs.abs()).sum())
    total = int(difference.numel())
    print(
        f"[parity] {what}: shape={tuple(ours.shape)} dtypes={dtypes} "
        f"tol=({atol:g},{rtol:g}) "
        f"mismatched={mismatched}/{total} ({100.0 * mismatched / total:.1f}%) "
        f"max_abs={float(difference.max()):.3e} "
        f"max_rel={float((difference / theirs.abs().clamp_min(1e-6)).max()):.3e} "
        f"mean_abs={float(difference.mean()):.3e}",
        flush=True,
    )
    torch.testing.assert_close(ours, theirs, atol=atol, rtol=rtol, msg=what)


@pytest.fixture(scope="module")
def parity_models():
    source = os.environ.get("VOXCPM2_UPSTREAM_SOURCE")
    checkpoint = os.environ.get("VOXCPM2_PARITY_CHECKPOINT")
    if not source or not checkpoint:
        pytest.skip(
            "Set VOXCPM2_UPSTREAM_SOURCE and VOXCPM2_PARITY_CHECKPOINT "
            "for real checkpoint parity"
        )
    if not torch.cuda.is_available():
        pytest.skip("VoxCPM2 checkpoint parity requires CUDA")

    sys.path.insert(0, str(Path(source) / "src"))
    from voxcpm.model.voxcpm2 import VoxCPM2Model

    from sglang_omni.models.voxcpm2.hf_config import load_voxcpm2_config
    from sglang_omni.models.voxcpm2.stages import _load_audio_vae
    from sglang_omni.utils.checkpoint import resolve_checkpoint

    checkpoint = resolve_checkpoint(checkpoint)
    config = load_voxcpm2_config(checkpoint)
    upstream = VoxCPM2Model.from_local(checkpoint, device="cuda:0")
    upstream.eval()

    ours = {
        "audio_vae": _load_audio_vae(checkpoint, config, device="cuda:0"),
        "config": config,
        "checkpoint": checkpoint,
    }
    return upstream, ours


def test_audio_vae_encode_matches_upstream(parity_models):
    upstream, ours = parity_models
    sample_rate = ours["config"].sample_rate
    waveform = _seeded(1, 1, sample_rate)

    with torch.inference_mode():
        theirs = upstream.audio_vae.encode(waveform, sample_rate)
        mine = ours["audio_vae"].encode(waveform, sample_rate)
    _assert_close(mine, theirs, "AudioVAE encode")


def test_audio_vae_decode_matches_upstream(parity_models):
    upstream, ours = parity_models
    latent_dim = ours["config"].latent_dim
    latents = _seeded(1, latent_dim, 32)

    with torch.inference_mode():
        theirs = upstream.audio_vae.decode(latents)
        mine = ours["audio_vae"].decode(latents)
    _assert_close(mine, theirs, "AudioVAE decode")


def test_local_encoder_matches_upstream(parity_models):
    upstream, ours = parity_models
    config = ours["config"]
    dtype = next(upstream.feat_encoder.parameters()).dtype
    patches = _seeded(1, 3, config.patch_size, config.feat_dim).to(dtype)

    with torch.inference_mode():
        theirs = upstream.feat_encoder(patches)
        mine = _our_engine_module(ours, "feat_encoder")(patches)
    _assert_close(mine, theirs, "local encoder")


def test_local_dit_matches_upstream(parity_models):
    upstream, ours = parity_models
    config = ours["config"]
    dtype = next(upstream.feat_decoder.parameters()).dtype
    dit_hidden = config.dit["hidden_dim"] * 2

    x = _seeded(1, config.feat_dim, config.patch_size, dtype=dtype)
    mu = _seeded(1, dit_hidden, dtype=dtype)
    cond = _seeded(1, config.feat_dim, config.patch_size, dtype=dtype)
    t = torch.full((1,), 0.5, device="cuda:0", dtype=dtype)
    dt = torch.zeros_like(t)

    with torch.inference_mode():
        theirs = upstream.feat_decoder.estimator(x, mu, t, cond, dt)
        mine = _our_engine_module(ours, "feat_decoder").estimator(x, mu, t, cond, dt)
    _assert_close(mine, theirs, "local DiT")


def test_flow_sampler_matches_upstream(parity_models):
    """Same seed, same recipe: the euler path must land on the same latents."""
    upstream, ours = parity_models
    config = ours["config"]
    dtype = next(upstream.feat_decoder.parameters()).dtype
    dit_hidden = config.dit["hidden_dim"] * 2

    mu = _seeded(1, dit_hidden, dtype=dtype)
    cond = _seeded(1, config.feat_dim, config.patch_size, dtype=dtype)

    with torch.inference_mode():
        torch.manual_seed(_SEED)
        theirs = upstream.feat_decoder(
            mu=mu,
            n_timesteps=10,
            patch_size=config.patch_size,
            cond=cond,
            cfg_value=2.0,
        )
        torch.manual_seed(_SEED)
        mine = _our_engine_module(ours, "feat_decoder")(
            mu=mu,
            n_timesteps=10,
            patch_size=config.patch_size,
            cond=cond,
            cfg_value=2.0,
        )
    _assert_close(mine, theirs, "flow-matching sampler")


def _our_engine_module(ours: dict, name: str):
    """Build the AR model's ported submodules without standing up SGLang."""
    cached = ours.get(f"_{name}")
    if cached is not None:
        return cached

    from sglang_omni.models.weight_loader import load_module

    config = ours["config"]
    if name == "feat_encoder":
        from sglang_omni.models.voxcpm2.components.local_encoder import VoxCPMLocEnc
        from sglang_omni.models.voxcpm2.sglang_model import _local_config

        module = VoxCPMLocEnc(
            _local_config(_lm_namespace(config), config.encoder),
            input_dim=config.feat_dim,
        )
    else:
        from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM
        from sglang_omni.models.voxcpm2.components.local_dit import VoxCPMLocDiT
        from sglang_omni.models.voxcpm2.sglang_model import _local_config

        module = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=CfmConfig(**config.cfm),
            estimator=VoxCPMLocDiT(
                _local_config(_lm_namespace(config), config.dit),
                in_channels=config.feat_dim,
            ),
            mean_mode=config.dit_mean_mode,
        )

    module = load_module(module, ours["checkpoint"], prefix=f"{name}.", device="cuda:0")
    ours[f"_{name}"] = module
    return module


def _lm_namespace(config):
    from types import SimpleNamespace

    return SimpleNamespace(**config.lm)
