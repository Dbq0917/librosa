#!/usr/bin/env python
"""Compatibility tests for the layered exception hierarchy."""

import numpy as np
import pytest

import librosa
import librosa.core.audio as audio
from librosa.util.exceptions import processing_errors


def test_exception_hierarchy():
    assert issubclass(librosa.ParameterError, librosa.LibrosaError)
    assert issubclass(librosa.ParameterError, librosa.InputError)
    assert issubclass(librosa.ParameterError, librosa.LibrosaParameterError)
    assert issubclass(librosa.ParameterError, ValueError)
    assert issubclass(librosa.ParameterError, TypeError)

    assert issubclass(librosa.ProcessingError, librosa.LibrosaError)
    assert issubclass(librosa.ProcessingError, librosa.LibrosaProcessingError)
    assert issubclass(librosa.ProcessingError, RuntimeError)
    assert issubclass(librosa.ProcessingError, OSError)


@pytest.mark.parametrize(
    "error_type",
    [
        librosa.ParameterError,
        librosa.LibrosaParameterError,
        librosa.InputError,
        ValueError,
        librosa.LibrosaError,
    ],
)
def test_old_and_new_parameter_types_share_handler(error_type):
    with pytest.raises(error_type):
        raise librosa.ParameterError(
            "sr must be strictly positive",
            parameter="sr",
            value=0,
        )


@pytest.mark.parametrize(
    "error_type",
    [
        librosa.ProcessingError,
        librosa.LibrosaProcessingError,
        RuntimeError,
        librosa.LibrosaError,
    ],
)
def test_old_and_new_processing_types_share_handler(error_type):
    with pytest.raises(error_type):
        raise librosa.ProcessingError(
            "audio decoding failed",
            operation="decode audio",
        )


def test_parameter_error_reports_name_and_layer():
    with pytest.raises(librosa.ParameterError) as exc_info:
        librosa.load("tests/test_audio.ogg", sr=0)

    exc = exc_info.value
    assert exc.layer == "parameter"
    assert exc.parameter == "sr"
    assert exc.value == 0
    assert "parameter=sr" in str(exc)
    assert isinstance(exc, ValueError)


def test_processing_error_preserves_backend_cause(monkeypatch):
    def fail_resample(*args, **kwargs):
        raise OSError("backend unavailable")

    monkeypatch.setattr(audio.soxr, "resample", fail_resample)

    y = np.ones(100, dtype=np.float32)
    with pytest.raises(librosa.ProcessingError) as exc_info:
        librosa.resample(y, orig_sr=22050, target_sr=11025)

    exc = exc_info.value
    assert exc.layer == "processing"
    assert exc.operation == "soxr resampling"
    assert isinstance(exc.__cause__, OSError)
    assert isinstance(exc, RuntimeError)
    assert "[processing operation=soxr resampling]" in str(exc)


def test_missing_audio_source_is_processing_error(tmp_path):
    path = tmp_path / "missing.wav"

    with pytest.raises(librosa.ProcessingError) as exc_info:
        librosa.load(path)

    exc = exc_info.value
    assert exc.layer == "processing"
    assert exc.operation == "open audio source"
    assert exc.__cause__ is not None
    assert isinstance(exc, RuntimeError)


def test_processing_error_helper_does_not_wrap_parameter_errors():
    with pytest.raises(librosa.ParameterError, match="parameter=frame_length"):
        with processing_errors("frame analysis"):
            raise librosa.ParameterError(
                "frame_length must be positive",
                parameter="frame_length",
            )
