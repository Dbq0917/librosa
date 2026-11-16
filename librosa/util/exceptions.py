#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosa.

The exception hierarchy separates invalid caller input from failures that
happen while processing valid input::

    LibrosaError
    ├── InputError (ValueError, TypeError)
    │   └── LibrosaParameterError
    │       └── ParameterError
    └── LibrosaProcessingError (RuntimeError, OSError)
        └── ProcessingError

The historical names ``LibrosaParameterError`` and ``ParameterError`` remain
available.  ``ParameterError`` also derives from :class:`ValueError` and
:class:`TypeError`, and ``ProcessingError`` derives from both
:class:`RuntimeError` and :class:`OSError`, so callers can choose the level of
specificity that suits them without changing existing handlers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


_UNSET = object()


class LibrosaError(Exception):
    """The root librosa exception class.

    Attributes
    ----------
    layer
        A stable, human-readable category (for example ``"parameter"`` or
        ``"processing"``).  This is also included in ``str(error)`` so log
        output identifies the failing layer without requiring source lookup.
    parameter
        The offending parameter name for input-layer errors, if known.
    operation
        A short description of the processing operation that failed, if known.
    value
        The offending parameter value, if supplied.
    """

    layer = "librosa"

    def __init__(
        self,
        message: str = "",
        *args: Any,
        parameter: str | None = None,
        operation: str | None = None,
        value: Any = _UNSET,
    ) -> None:
        super().__init__(message, *args)
        self.parameter = parameter
        self.param = parameter
        self.operation = operation
        self.value = None if value is _UNSET else value

    @property
    def context(self) -> dict[str, Any]:
        """Return structured context suitable for structured logging."""
        return {
            "layer": self.layer,
            "parameter": self.parameter,
            "operation": self.operation,
            "value": self.value,
        }

    def __str__(self) -> str:
        message = super().__str__()
        details = [self.layer]
        if self.parameter is not None:
            details.append(f"parameter={self.parameter}")
        if self.operation is not None:
            details.append(f"operation={self.operation}")

        prefix = f"[{' '.join(details)}]"
        return f"{prefix} {message}" if message else prefix


class InputError(LibrosaError, ValueError, TypeError):
    """Base class for invalid values supplied by the caller."""

    layer = "parameter"


class LibrosaParameterError(InputError):
    """Historical parameter-error name retained for compatibility.

    New code should catch :class:`ParameterError`, :class:`InputError`,
    :class:`ValueError`, or :class:`TypeError`.  Existing handlers for this
    name continue to work because :class:`ParameterError` inherits from it.
    """


class ParameterError(LibrosaParameterError):
    """Exception class for malformed inputs or invalid parameter values.

    ``ParameterError`` remains the concrete exception raised by the library.
    It is also a :class:`ValueError`, :class:`TypeError`, and
    :class:`InputError`, allowing callers to catch the narrow historical name
    or the broader input layer.

    Use the ``parameter`` keyword when raising this exception::

        raise ParameterError(
            "sr must be strictly positive",
            parameter="sr",
            value=sr,
        )
    """


class LibrosaProcessingError(LibrosaError, RuntimeError, OSError):
    """Base class for failures while processing otherwise valid input."""

    layer = "processing"


class ProcessingError(LibrosaProcessingError):
    """Exception class for runtime processing failures.

    This type is used when a decoder, resampler, numerical backend, or another
    processing stage fails.  Low-level exceptions are wrapped at the boundary
    where they occur, and the original exception is retained as ``__cause__``::

        try:
            decoded = decoder.read()
        except DecoderFailure as exc:
            raise ProcessingError(
                "audio decoding failed",
                operation="decode audio",
            ) from exc
    """


@contextmanager
def processing_errors(operation: str) -> Iterator[None]:
    """Translate non-librosa exceptions raised by a processing stage.

    Parameters
    ----------
    operation
        Human-readable description of the narrow operation being wrapped.

    Yields
    ------
    None
        The wrapped processing block.

    Notes
    -----
    ``ParameterError`` and other :class:`LibrosaError` instances pass through
    unchanged.  Every other exception is re-raised as :class:`ProcessingError`
    with the original exception attached via ``raise ... from exc``, preserving
    both the traceback and the root cause for diagnostics.

    This helper is intended for narrow external-library or low-level processing
    boundaries, not as a blanket wrapper around an entire public API.
    """
    try:
        yield
    except LibrosaError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"{operation} failed: {exc}",
            operation=operation,
        ) from exc


__all__ = [
    "InputError",
    "LibrosaError",
    "LibrosaParameterError",
    "LibrosaProcessingError",
    "ParameterError",
    "ProcessingError",
    "processing_errors",
]
