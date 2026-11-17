#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the persistent feature-extraction cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pytest

import librosa
from librosa._cache import CacheManager, cache
from librosa.core.spectrum import _stft_from_cache, stft


def _cache_settings(manager: CacheManager) -> dict[str, object]:
    return {
        "cache_dir": manager.cache_dir,
        "version": manager.version,
        "max_bytes": manager.max_bytes,
        "max_items": manager.max_items,
        "memory_bytes": manager.memory_bytes,
        "block_frames": manager.block_frames,
        "level": manager.level,
    }


def _restore_settings(manager: CacheManager, settings: dict[str, object]) -> None:
    manager.configure(
        cache_dir=settings["cache_dir"],
        version=settings["version"],
        max_bytes=settings["max_bytes"],
        max_items=settings["max_items"],
        memory_bytes=settings["memory_bytes"],
        block_frames=settings["block_frames"],
        level=settings["level"],
    )


@contextmanager
def _disabled(manager: CacheManager):
    settings = _cache_settings(manager)
    manager.configure(cache_dir=None)
    try:
        yield
    finally:
        _restore_settings(manager, settings)


@pytest.fixture
def configured_cache(tmp_path):
    """Configure the global cache in a temporary directory."""
    settings = _cache_settings(cache)
    cache.configure(
        cache_dir=tmp_path,
        version="feature-cache-test",
        max_bytes=64 * 1024**2,
        max_items=2048,
        memory_bytes=16 * 1024**2,
        block_frames=16,
        level=40,
    )
    cache.clear()
    cache.reset_stats()
    try:
        yield cache, tmp_path
    finally:
        _restore_settings(cache, settings)


def _fresh_mfcc(y, sr, **kwargs):
    with _disabled(cache):
        return librosa.feature.mfcc.__wrapped__(y=y, sr=sr, **kwargs)


def test_cached_mfcc_is_exact_and_reuses_stages(configured_cache):
    """Cached MFCCs must be exact and reuse unaffected pipeline stages."""
    manager, _ = configured_cache
    rng = np.random.default_rng(440)
    sr = 8000
    y = rng.standard_normal(3 * sr).astype(np.float32)

    cases = [
        {"n_mfcc": 20, "n_mels": 64, "n_fft": 256, "hop_length": 64},
        {"n_mfcc": 24, "n_mels": 64, "n_fft": 256, "hop_length": 64},
        {"n_mfcc": 24, "n_mels": 64, "n_fft": 256, "hop_length": 64, "lifter": 10},
        {"n_mfcc": 24, "n_mels": 96, "n_fft": 256, "hop_length": 64, "lifter": 10},
    ]
    results = []
    timings = []

    for params in cases:
        start = time.perf_counter()
        results.append(librosa.feature.mfcc(y=y, sr=sr, **params))
        timings.append(time.perf_counter() - start)

    for params, result in zip(cases, results, strict=True):
        baseline = _fresh_mfcc(y, sr, **params)
        assert np.array_equal(result, baseline)
        assert np.max(np.abs(result - baseline)) == 0

    # The same parameters should be served from the function cache.
    manager.reset_stats()
    repeated = librosa.feature.mfcc(y=y, sr=sr, **cases[-1])
    assert np.array_equal(repeated, results[-1])
    assert manager.stats["hits"] >= 1

    # Changing only an MFCC parameter should not touch the STFT blocks.
    manager.reset_stats()
    _ = librosa.feature.mfcc(y=y, sr=sr, **{**cases[-1], "n_mfcc": 12})
    by_stage = manager.stats["by_stage"]
    assert by_stage["librosa.feature.spectral.mfcc"]["misses"] == 1
    assert by_stage["librosa.feature.spectral.melspectrogram"]["hits"] == 1
    assert "librosa.core.spectrum.stft.blocks" not in by_stage

    # Changing a mel parameter should reuse the STFT blocks and recompute
    # only the mel/MFCC stages.
    manager.reset_stats()
    _ = librosa.feature.mfcc(y=y, sr=sr, **{**cases[-1], "fmax": 3000})
    by_stage = manager.stats["by_stage"]
    assert by_stage["librosa.core.spectrum.stft.blocks"]["misses"] == 0
    assert by_stage["librosa.core.spectrum.stft.blocks"]["hits"] > 0

    assert timings[1] < timings[0]
    assert timings[2] < timings[0]


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["constant", "edge", "reflect"])
@pytest.mark.parametrize("channels", [1, 2])
def test_block_stft_matches_full_stft(configured_cache, center, pad_mode, channels):
    """Blockwise STFT must match the full STFT bit for bit."""
    manager, _ = configured_cache
    manager.configure(block_frames=7)
    rng = np.random.default_rng(1234)
    n_fft = 512
    hop_length = 128
    shape = (channels, 5000) if channels > 1 else (5000,)
    y = rng.standard_normal(shape).astype(np.float32)

    full = getattr(stft, "__wrapped__", stft)(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
        pad_mode=pad_mode,
    )
    blocked = _stft_from_cache(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=center,
        pad_mode=pad_mode,
    )
    assert np.array_equal(full, blocked)
    assert np.max(np.abs(full - blocked)) == 0

    blocked_magnitude = _stft_from_cache(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=center,
        pad_mode=pad_mode,
        magnitude=True,
    )
    assert np.array_equal(np.abs(full), blocked_magnitude)

    # Re-run from a cold process-like memory cache.  The disk values must
    # still reconstruct exactly the same result.
    manager.configure(cache_dir=manager.cache_dir, block_frames=7)
    blocked_again = _stft_from_cache(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=center,
        pad_mode=pad_mode,
    )
    assert np.array_equal(full, blocked_again)


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_cached_mel_power_is_exact(configured_cache, power):
    """Changing the spectrogram exponent must remain bit-exact."""
    _ = configured_cache
    rng = np.random.default_rng(99)
    y = rng.standard_normal(6000).astype(np.float32)
    cached = librosa.feature.melspectrogram(
        y=y,
        sr=8000,
        n_fft=256,
        hop_length=64,
        n_mels=32,
        power=power,
    )
    with _disabled(cache):
        baseline = librosa.feature.melspectrogram.__wrapped__(
            y=y,
            sr=8000,
            n_fft=256,
            hop_length=64,
            n_mels=32,
            power=power,
        )
    assert np.array_equal(cached, baseline)
    assert np.max(np.abs(cached - baseline)) == 0


def test_function_cache_key_covers_kwargs_and_array_content(tmp_path):
    """Function keys must include keyword arguments and array contents."""
    manager = CacheManager(
        tmp_path,
        level=40,
        max_bytes=8 * 1024**2,
        memory_bytes=2 * 1024**2,
    )
    calls = []

    @manager(level=-10)
    def compute(value, scale=1, **kwargs):
        calls.append(kwargs)
        return value * scale + kwargs.get("offset", 0)

    value = np.arange(8, dtype=np.float64)
    assert np.array_equal(compute(value, scale=2, offset=1), value * 2 + 1)
    assert np.array_equal(compute(value, scale=2, offset=1), value * 2 + 1)
    assert len(calls) == 1

    assert np.array_equal(compute(value, scale=2, offset=2), value * 2 + 2)
    assert len(calls) == 2

    value[0] = 100
    assert np.array_equal(compute(value, scale=2, offset=1), value * 2 + 1)
    assert len(calls) == 3


def test_cache_lru_limits_prefix_and_version(tmp_path):
    """The cache must enforce LRU limits and prefix/version invalidation."""
    manager = CacheManager(
        tmp_path,
        version="v1",
        max_bytes=12000,
        max_items=8,
        memory_bytes=3500,
        level=40,
    )
    keys = [manager.key("mel", n_mels=n, fmax=8000) for n in (32, 64, 128, 256)]
    for idx, key in enumerate(keys):
        manager.put(key, np.full(256, idx, dtype=np.float64))

    assert manager.stats["disk_bytes"] <= manager.max_bytes
    assert manager.stats["memory_bytes"] <= manager.memory_bytes

    # Touch the first entry, then force eviction.  The recently-used entry
    # should survive instead of the least-recently-used one.
    manager.get(keys[0])
    for idx in range(8):
        manager.put(
            manager.key("mel", n_mels=512 + idx, fmax=8000),
            np.full(256, idx, dtype=np.float64),
        )
        if manager.stats["evictions"] > 0:
            break
    assert manager.stats["evictions"] > 0
    assert manager.contains(keys[0])

    manager.put(manager.key("mfcc", n_mfcc=20, n_mels=64), np.ones(16))
    manager.clear_prefix(("mel", "n_mels"))
    assert all(not manager.contains(key) for key in keys)
    assert manager.get(manager.key("mfcc", n_mfcc=20, n_mels=64)) is not None

    survivor = manager.key("mel", n_mels=64, fmax=8000)
    manager.put(survivor, np.arange(32))
    assert manager.contains(survivor)
    manager.configure(version="v2")
    assert not manager.contains(survivor)
    assert manager.stats["disk_items"] == 0


def test_corrupt_files_and_index_are_rebuilt(configured_cache):
    """Corrupted data files and indexes must be rebuilt transparently."""
    manager, cache_root = configured_cache
    rng = np.random.default_rng(7)
    y = rng.standard_normal(4000).astype(np.float32)
    params = {"n_mfcc": 13, "n_fft": 256, "hop_length": 64, "n_mels": 32}
    expected = librosa.feature.mfcc(y=y, sr=8000, **params)

    files = list((cache_root / "librosa-block-cache" / "data").glob("**/*.pkl"))
    assert files
    for path in files:
        path.write_bytes(b"not a valid cache entry")

    manager.configure(cache_dir=manager.cache_dir, block_frames=16)
    manager.reset_stats()
    rebuilt = librosa.feature.mfcc(y=y, sr=8000, **params)
    assert np.array_equal(rebuilt, expected)
    assert manager.stats["corruptions"] > 0

    db_path = cache_root / "librosa-block-cache" / "index.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        path = db_path.with_name(db_path.name + suffix)
        if suffix == "":
            path.write_bytes(b"corrupt")
        elif path.exists():
            path.unlink()

    manager.configure(cache_dir=manager.cache_dir, block_frames=16)
    rebuilt = librosa.feature.mfcc(y=y, sr=8000, **params)
    assert np.array_equal(rebuilt, expected)


def test_cache_is_reused_across_processes(tmp_path):
    """A fresh process must be able to reuse the persisted cache."""
    env = os.environ.copy()
    env.update(
        {
            "LIBROSA_CACHE_DIR": str(tmp_path),
            "LIBROSA_CACHE_VERSION": "cross-process",
            "LIBROSA_CACHE_LEVEL": "40",
            "LIBROSA_CACHE_BLOCK_FRAMES": "8",
            "LIBROSA_CACHE_MAX_BYTES": "64M",
            "LIBROSA_CACHE_MEMORY_BYTES": "8M",
        }
    )
    script = """
import json
import numpy as np
import librosa

y = np.linspace(-1, 1, 6000, dtype=np.float32)
out = librosa.feature.mfcc(
    y=y, sr=8000, n_mfcc=13, n_fft=256, hop_length=64, n_mels=32
)
print(json.dumps({"sum": float(out.sum()), "stats": librosa.cache.stats}))
"""

    first = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    first_result = json.loads(first.stdout.strip().splitlines()[-1])
    second_result = json.loads(second.stdout.strip().splitlines()[-1])
    assert first_result["sum"] == second_result["sum"]
    assert second_result["stats"]["disk_hits"] > 0
