Caching
^^^^^^^

This section covers the *librosa* function cache.  This allows you
to store and reuse intermediate computations across sessions.

Enabling the cache
------------------
By default, caching is disabled.  To enable caching, the environment 
variable `LIBROSA_CACHE_DIR` must be set prior to loading *librosa*.
This can be done on the command line prior to instantiating a python interpreter::

    $ export LIBROSA_CACHE_DIR=/tmp/librosa_cache
    $ ipython

or from within python, prior to importing *librosa*::

    >>> import os
    >>> os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
    >>> import librosa

Cache configuration
-------------------
The cache stores both whole-function results and intermediate feature
blocks.  Entries are persisted on disk, can be reused by other processes,
and are indexed for least-recently-used (LRU) eviction.

The default configuration can be overridden by setting the following
environment variables before importing *librosa*:

  - `LIBROSA_CACHE_DIR` : path (on disk) to the cache directory
  - `LIBROSA_CACHE_MMAP` : optional memory mapping mode `{None, 'r+', 'r', 'w+', 'c'}`
  - `LIBROSA_CACHE_COMPRESS` : flag to enable compression of data on disk `{0, 1}`
  - `LIBROSA_CACHE_VERBOSE` : controls how much debug info is displayed. `{int, non-negative}`
  - `LIBROSA_CACHE_LEVEL` : controls the caching level: the larger this value, the more data is cached. `{int}`
  - `LIBROSA_CACHE_MAX_BYTES` : maximum on-disk data size. Size suffixes such as `512M` are accepted.
  - `LIBROSA_CACHE_MEMORY_BYTES` : maximum in-process memory used by the cache.
  - `LIBROSA_CACHE_MAX_ITEMS` : maximum number of disk entries.
  - `LIBROSA_CACHE_BLOCK_FRAMES` : number of frames per intermediate block (default: 512).
  - `LIBROSA_CACHE_VERSION` : cache version tag. Changing this invalidates old entries automatically.

The cache can also be configured at runtime::

    >>> librosa.cache.configure(
    ...     cache_dir="/mnt/audio-cache",
    ...     max_bytes="20G",
    ...     memory_bytes="512M",
    ...     block_frames=1024,
    ... )

The cache is disabled when `cache_dir` is `None`.

As of 0.7, librosa's cache wraps (rather than extends) a `joblib.Memory`
object for backwards compatibility.  The memory object can still be
accessed directly by `librosa.cache.memory`; new cache entries use the
bounded block cache.


Feature blocks and parameter reuse
----------------------------------
Feature extraction caches intermediate magnitude-spectrogram blocks
derived from the STFT.  If a parameter only
affects a downstream stage, earlier stages are reused:

  - changing an MFCC parameter such as `n_mfcc` or `lifter` reuses the
    Mel and STFT results
  - changing a Mel parameter such as `n_mels` or `fmax` reuses the STFT
    blocks
  - changing an STFT parameter such as `hop_length` or `n_fft`
    invalidates only the spectrogram blocks that depend on it

Cached values are bit-for-bit identical to values computed without the
cache.


Invalidation and reporting
--------------------------
Individual parameter groups can be invalidated without touching unrelated
entries::

    >>> librosa.cache.clear_param("n_mels")
    >>> librosa.cache.clear_prefix("librosa.feature.spectral.melspectrogram")
    >>> librosa.cache.clear_prefix(("stft", "hop_length"))

Changing `LIBROSA_CACHE_VERSION` (or passing `version=` to `configure`)
automatically discards entries from older versions.

Corrupted entries are deleted and recomputed on the next access.  A
corrupted disk index is rebuilt automatically.

Cache hit and miss statistics can be reported for monitoring::

    >>> stats = librosa.cache.stats
    >>> stats["hit_rate"]
    0.75
    >>> stats["by_stage"]
    {'librosa.core.spectrum.stft.blocks': {...}, ...}

Statistics include memory and disk hits, misses, writes, evictions,
corruption recoveries, current memory/disk usage, and per-stage hit rates.


Cache levels
------------

Cache levels operate in a fashion similar to logging levels.
For small values of `LIBROSA_CACHE_LEVEL`, only the most important (frequently used) data are cached.
As the cache level increases, broader classes of functions are cached.
As a result, application code may run faster at the expense of larger disk usage.

The caching levels are described as follows:

    - 10: filter bases, independent of audio data (mel, chroma, constant-q)
    - 20: low-level features (cqt, stft, zero-crossings, etc)
    - 30: high-level features (tempo, beats, decomposition, recurrence, etc)
    - 40: post-processing (delta, stack_memory, normalize, sync)

The default cache level is 10.
