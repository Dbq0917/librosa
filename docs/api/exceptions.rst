Exceptions
==========

librosa separates failures caused by caller input from failures that occur
while processing valid input.  Both categories inherit from
:class:`librosa.LibrosaError`, so existing top-level handlers continue to work.

.. code-block:: text

    LibrosaError
    ├── InputError (ValueError, TypeError)
    │   └── LibrosaParameterError
    │       └── ParameterError
    └── LibrosaProcessingError (RuntimeError, OSError)
        └── ProcessingError

.. autoclass:: librosa.LibrosaError
    :members:

.. autoclass:: librosa.InputError

.. autoclass:: librosa.LibrosaParameterError

.. autoclass:: librosa.ParameterError

.. autoclass:: librosa.LibrosaProcessingError

.. autoclass:: librosa.ProcessingError

.. autofunction:: librosa.util.exceptions.processing_errors

Choosing a handler
------------------

Use :class:`librosa.ParameterError` for bad arguments, and
:class:`librosa.ProcessingError` for failures from decoding, resampling, or
other processing backends.  If a service reports these categories differently,
catch the two categories explicitly:

.. code-block:: python

    import logging
    import librosa

    logger = logging.getLogger(__name__)

    try:
        y, sr = librosa.load(path, sr=sample_rate)
    except librosa.ParameterError as exc:
        logger.error(
            "invalid request: layer=%s parameter=%s",
            exc.layer,
            exc.parameter,
        )
    except librosa.ProcessingError as exc:
        logger.exception(
            "audio processing failed: layer=%s operation=%s",
            exc.layer,
            exc.operation,
        )

Every librosa exception includes its layer in ``str(exc)``.  Input-layer
messages also include the offending parameter name whenever it is available,
and the ``parameter``, ``operation``, and ``value`` attributes can be emitted
as structured log fields.

Compatibility map
-----------------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - Existing usage
      - Preferred usage
      - Compatibility
    * - ``except LibrosaError``
      - Unchanged
      - Catches both parameter and processing errors.
    * - ``except ParameterError``
      - Unchanged, or ``except InputError``
      - Unchanged; also catches as ``ValueError``, ``TypeError``, and ``InputError``.
    * - ``except LibrosaParameterError``
      - ``except ParameterError``
      - Still catches ``ParameterError`` through inheritance.
    * - Raw ``ValueError`` from a parameter check
      - ``ParameterError`` with ``parameter=...``
      - ``ValueError`` and ``TypeError`` handlers continue to work.
    * - Raw decoder/resampler ``RuntimeError``/``OSError``
      - ``ProcessingError`` with ``operation=...``
      - Wrap with ``raise ... from exc`` at the narrow processing boundary.
    * - ``except RuntimeError``
      - ``except ProcessingError`` when the category matters
      - ``ProcessingError`` still derives from ``RuntimeError`` and ``OSError``.

The original low-level exception is retained as ``__cause__`` when a processing
failure is translated.  The traceback is therefore available to debuggers and
log formatters, while callers no longer need broad handlers for external
backend exceptions.
