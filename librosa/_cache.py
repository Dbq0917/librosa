#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function and intermediate-result caching.

The cache is deliberately conservative: it is disabled unless a cache
directory is configured, and every cache operation is best-effort.  A
missing, unreadable, or corrupted cache entry is treated as a miss so that
audio analysis can continue and rebuild the entry.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import os
import pickle
import re
import sqlite3
import struct
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
from joblib import Memory

from .version import version as _librosa_version

__all__ = [
    "BlockCache",
    "CacheKey",
    "CacheManager",
    "CacheStats",
    "cache",
    "cache_key",
]


_F = TypeVar("_F", bound=Callable[..., Any])

# Bump this whenever the layout or semantics of cached values changes.
_CACHE_FORMAT_VERSION = 2
_CACHE_ROOT_NAME = "librosa-block-cache"
_UNSET = object()
_DIGEST_SCOPE: contextvars.ContextVar[
    dict[int, tuple[np.ndarray, bytes]] | None
] = contextvars.ContextVar("librosa_cache_digest_scope", default=None)


def _parse_bytes(value: Any, *, default: int | None = None) -> int | None:
    """Parse a byte count, accepting common size suffixes."""
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgtpe]?i?b?)?", text, re.I)
        if match is None:
            raise ValueError(f"Invalid byte size: {value!r}")
        number = float(match.group(1))
        suffix = (match.group(2) or "").lower().rstrip("b")
        factors = {
            "": 1,
            "k": 1024,
            "ki": 1024,
            "m": 1024**2,
            "mi": 1024**2,
            "g": 1024**3,
            "gi": 1024**3,
            "t": 1024**4,
            "ti": 1024**4,
            "p": 1024**5,
            "pi": 1024**5,
            "e": 1024**6,
            "ei": 1024**6,
        }
        return max(0, int(number * factors.get(suffix, 1)))
    return max(0, int(value))


def _parse_bool(value: Any) -> bool:
    """Parse a boolean environment variable."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _nbytes(value: Any) -> int:
    """Estimate the resident size of a cached value."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.generic):
        return int(value.nbytes)
    if isinstance(value, dict):
        return int(
            sys.getsizeof(value)
            + sum(_nbytes(key) + _nbytes(val) for key, val in value.items())
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return int(sys.getsizeof(value) + sum(_nbytes(item) for item in value))
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return int(sys.getsizeof(value))
    return int(sys.getsizeof(value))


def _copy_value(value: Any) -> Any:
    """Return a defensive copy when possible."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _hash_array(value: np.ndarray, digest: "hashlib._Hash") -> None:
    digest.update(b"ndarray")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(struct.pack("<I", value.ndim))
    for size in value.shape:
        digest.update(struct.pack("<q", int(size)))
    for stride in value.strides:
        digest.update(struct.pack("<q", int(stride)))
    digest.update(b"F" if value.flags.f_contiguous else b"C")
    if not value.flags.c_contiguous and not value.flags.f_contiguous:
        digest.update(np.ascontiguousarray(value).tobytes())
    else:
        digest.update(memoryview(np.ravel(value, order="K")).cast("B"))


def _hash_object(value: Any, digest: "hashlib._Hash") -> None:
    """Update a digest with a deterministic representation of ``value``."""
    if isinstance(value, np.ndarray):
        _hash_array(value, digest)
        return
    if isinstance(value, np.generic):
        digest.update(b"npscalar")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
        return
    if value is None:
        digest.update(b"none")
        return
    if isinstance(value, bool):
        digest.update(b"bool1" if value else b"bool0")
        return
    if isinstance(value, int):
        digest.update(b"int")
        digest.update(str(value).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float")
        digest.update(struct.pack(">d", value))
        return
    if isinstance(value, complex):
        digest.update(b"complex")
        digest.update(struct.pack(">dd", value.real, value.imag))
        return
    if isinstance(value, str):
        digest.update(b"str")
        digest.update(value.encode("utf-8", "surrogatepass"))
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"bytes")
        digest.update(bytes(value))
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        items = [(_digest_value(key), val) for key, val in value.items()]
        digest.update(struct.pack("<q", len(items)))
        for key_digest, val in sorted(items, key=lambda item: item[0]):
            digest.update(key_digest)
            _hash_object(val, digest)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list" if isinstance(value, list) else b"tuple")
        digest.update(struct.pack("<q", len(value)))
        for item in value:
            _hash_object(item, digest)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(b"set")
        item_digests = sorted(_digest_value(item) for item in value)
        digest.update(struct.pack("<q", len(item_digests)))
        for item_digest in item_digests:
            digest.update(item_digest)
        return
    if isinstance(value, type):
        digest.update(b"type")
        digest.update(
            f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', value)}".encode(
                "utf-8", "surrogatepass"
            )
        )
        return
    if callable(value):
        digest.update(b"callable")
        target = inspect.unwrap(value)
        digest.update(
            f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', target)}".encode(
                "utf-8", "surrogatepass"
            )
        )
        code = getattr(target, "__code__", None)
        if code is not None:
            digest.update(code.co_code)
            digest.update(repr(code.co_consts).encode("utf-8", "surrogatepass"))
        return

    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        payload = repr(value).encode("utf-8", "surrogatepass")
    digest.update(b"pickle")
    digest.update(payload)


def _digest_value(value: Any) -> bytes:
    """Return a stable, process-independent digest for an arbitrary value."""
    # SHA-256 uses hardware acceleration on common platforms and is much
    # faster than BLAKE2 for large audio arrays.
    scope = _DIGEST_SCOPE.get()
    if isinstance(value, np.ndarray) and value.nbytes >= 4096 and scope is not None:
        cached = scope.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
    digest = hashlib.sha256()
    _hash_object(value, digest)
    result = digest.digest()
    if isinstance(value, np.ndarray) and value.nbytes >= 4096 and scope is not None:
        scope[id(value)] = (value, result)
    return result


def _key_label(value: Any) -> str:
    """Return a compact, filesystem-friendly label for a key value."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)) or value is None:
        text = repr(value)
    else:
        text = type(value).__name__
    text = re.sub(r"[^A-Za-z0-9_.:=+-]+", "_", text)
    return text[:96]


@dataclass(frozen=True)
class CacheKey:
    """A structured cache key.

    Parameters
    ----------
    stage : str
        Name of the computation stage, e.g. ``"stft"``.
    params : tuple of (str, object)
        Named parameters that affect the cached value.
    """

    stage: str
    params: tuple[tuple[str, Any], ...] = ()


def cache_key(stage: str, /, **params: Any) -> CacheKey:
    """Construct a structured cache key from named parameters."""
    return CacheKey(stage=os.fspath(stage), params=tuple(params.items()))


@dataclass
class _Stats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    writes: int = 0
    evictions: int = 0
    corruptions: int = 0
    skipped: int = 0
    by_stage: dict[str, dict[str, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_stage is None:
            self.by_stage = defaultdict(
                lambda: {
                    "requests": 0,
                    "hits": 0,
                    "misses": 0,
                    "memory_hits": 0,
                    "disk_hits": 0,
                }
            )

    def hit(self, stage: str, source: str) -> None:
        self.requests += 1
        self.hits += 1
        stage_stats = self.by_stage[stage]
        stage_stats["requests"] += 1
        stage_stats["hits"] += 1
        if source == "memory":
            self.memory_hits += 1
            stage_stats["memory_hits"] += 1
        else:
            self.disk_hits += 1
            stage_stats["disk_hits"] += 1

    def miss(self, stage: str) -> None:
        self.requests += 1
        self.misses += 1
        stage_stats = self.by_stage[stage]
        stage_stats["requests"] += 1
        stage_stats["misses"] += 1


class CacheStats(dict):
    """Dictionary-like cache statistics that can also be called."""

    def __call__(self) -> "CacheStats":
        return self

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class _PreparedKey:
    stage: str
    params: tuple[tuple[str, Any], ...]
    digest: bytes
    path: str


class _DiskBackend:
    """SQLite-indexed, file-backed LRU cache."""

    def __init__(
        self,
        root: Path,
        *,
        version: str,
        max_bytes: int | None,
        max_items: int | None,
        stats: _Stats,
    ) -> None:
        self.root = root
        self.version = version
        self.max_bytes = max_bytes
        self.max_items = max_items
        self.stats = stats
        self.data_dir = self.root / "data"
        self.db_path = self.root / "index.sqlite3"
        self._lock = threading.RLock()
        self._disabled = False
        self._ensure()

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _ensure(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._remove_stale_temps()
            with self._connect() as con:
                self._purge_other_versions(con)
            self._evict()
        except (OSError, sqlite3.Error) as exc:
            warnings.warn(
                f"librosa cache disabled: unable to initialize {self.root}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._disabled = True

    def _remove_stale_temps(self) -> None:
        for path in self.root.glob(".tmp-*"):
            with contextlib.suppress(OSError):
                path.unlink()

    def _new_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                digest TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                path TEXT NOT NULL,
                labels TEXT NOT NULL,
                version TEXT NOT NULL,
                size INTEGER NOT NULL,
                created REAL NOT NULL,
                accessed REAL NOT NULL
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS entries_accessed ON entries(accessed)")
        con.execute("CREATE INDEX IF NOT EXISTS entries_version ON entries(version)")
        con.execute(
            "SELECT digest, stage, path, labels, version, size, created, accessed "
            "FROM entries LIMIT 0"
        )
        con.commit()
        return con

    def _reset_database(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            with contextlib.suppress(FileNotFoundError):
                Path(f"{self.db_path}{suffix}").unlink()
        try:
            for path in self.data_dir.glob("*"):
                if path.is_dir():
                    for child in path.iterdir():
                        child.unlink()
                    path.rmdir()
                else:
                    path.unlink()
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        existed = self.db_path.exists()
        if not existed:
            for suffix in ("-wal", "-shm", "-journal"):
                with contextlib.suppress(OSError):
                    Path(f"{self.db_path}{suffix}").unlink()
        try:
            con = self._new_connection()
            if not existed:
                self._clean_orphans(con)
            return con
        except sqlite3.DatabaseError:
            with self._lock:
                self._reset_database()
            return self._new_connection()

    def _clean_orphans(self, con: sqlite3.Connection) -> None:
        """Remove data files that have no corresponding index entry."""
        try:
            known = {
                row["digest"]
                for row in con.execute("SELECT digest FROM entries").fetchall()
            }
            for path in self.data_dir.glob("*/*.pkl"):
                if path.stem not in known:
                    with contextlib.suppress(OSError):
                        path.unlink()
        except (OSError, sqlite3.Error):
            pass

    def _purge_other_versions(self, con: sqlite3.Connection) -> None:
        rows = con.execute(
            "SELECT digest, version FROM entries WHERE version != ?", (self.version,)
        ).fetchall()
        if not rows:
            return
        for row in rows:
            self._unlink_digest(row["digest"])
        con.executemany(
            "DELETE FROM entries WHERE digest = ?",
            [(row["digest"],) for row in rows],
        )
        con.commit()

    def _file_path(self, digest: str) -> Path:
        return self.data_dir / digest[:2] / f"{digest}.pkl"

    def _unlink_digest(self, digest: str) -> None:
        path = self._file_path(digest)
        with contextlib.suppress(OSError):
            path.unlink()
        with contextlib.suppress(OSError):
            path.parent.rmdir()

    @staticmethod
    def _decode_labels(raw: str) -> list[str]:
        try:
            labels = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if isinstance(labels, list):
            return [str(label) for label in labels]
        return []

    def lookup(self, digest: str) -> tuple[bool, Any]:
        if self._disabled:
            return False, None
        with self._lock:
            try:
                with self._connect() as con:
                    row = con.execute(
                        "SELECT * FROM entries WHERE digest = ? AND version = ?",
                        (digest, self.version),
                    ).fetchone()
                    if row is None:
                        return False, None
                    try:
                        with self._file_path(digest).open("rb") as f:
                            payload = pickle.load(f)  # noqa: S301
                    except Exception:
                        self.stats.corruptions += 1
                        self._delete_row(con, digest)
                        con.commit()
                        return False, None
                    if not (
                        isinstance(payload, dict)
                        and payload.get("format") == _CACHE_FORMAT_VERSION
                        and payload.get("version") == self.version
                        and payload.get("digest") == digest
                    ):
                        self.stats.corruptions += 1
                        self._delete_row(con, digest)
                        con.commit()
                        return False, None
                    con.execute(
                        "UPDATE entries SET accessed = ? WHERE digest = ?",
                        (time.time(), digest),
                    )
                    con.commit()
                    return True, payload["value"]
            except (OSError, sqlite3.Error, EOFError):
                return False, None

    def put(self, digest: str, *, value: Any, stage: str, labels: Iterable[str]) -> None:
        if self._disabled:
            return
        path = self._file_path(digest)
        tmp = self.root / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
        payload = {
            "format": _CACHE_FORMAT_VERSION,
            "version": self.version,
            "digest": digest,
            "value": value,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            size = path.stat().st_size
        except (OSError, pickle.PickleError):
            with contextlib.suppress(OSError):
                tmp.unlink()
            return

        with self._lock:
            try:
                with self._connect() as con:
                    con.execute(
                        """
                        INSERT INTO entries
                            (digest, stage, path, labels, version, size, created, accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(digest) DO UPDATE SET
                            stage=excluded.stage,
                            path=excluded.path,
                            labels=excluded.labels,
                            version=excluded.version,
                            size=excluded.size,
                            accessed=excluded.accessed
                        """,
                        (
                            digest,
                            stage,
                            str(path),
                            json.dumps(list(labels)),
                            self.version,
                            size,
                            time.time(),
                            time.time(),
                        ),
                    )
                    con.commit()
            except (OSError, sqlite3.Error):
                self._unlink_digest(digest)
                return
            self._evict()

    @staticmethod
    def _delete_row(con: sqlite3.Connection, digest: str) -> None:
        con.execute("DELETE FROM entries WHERE digest = ?", (digest,))

    def touch(self, digest: str) -> None:
        """Update the LRU timestamp for an entry."""
        if self._disabled:
            return
        with self._lock:
            try:
                with self._connect() as con:
                    con.execute(
                        "UPDATE entries SET accessed = ? WHERE digest = ?",
                        (time.time(), digest),
                    )
                    con.commit()
            except (OSError, sqlite3.Error):
                pass

    def delete(self, digest: str) -> None:
        if self._disabled:
            return
        with self._lock:
            try:
                with self._connect() as con:
                    self._delete_row(con, digest)
                    con.commit()
            except (OSError, sqlite3.Error):
                pass
            self._unlink_digest(digest)

    def _evict(self) -> None:
        if self._disabled:
            return
        with self._lock:
            try:
                with self._connect() as con:
                    while True:
                        row = con.execute(
                            "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS total "
                            "FROM entries WHERE version = ?",
                            (self.version,),
                        ).fetchone()
                        over_items = (
                            self.max_items is not None and row["n"] > self.max_items
                        )
                        over_bytes = (
                            self.max_bytes is not None and row["total"] > self.max_bytes
                        )
                        if not over_items and not over_bytes:
                            break
                        victim = con.execute(
                            "SELECT digest FROM entries WHERE version = ? "
                            "ORDER BY accessed ASC, created ASC LIMIT 1",
                            (self.version,),
                        ).fetchone()
                        if victim is None:
                            break
                        self._delete_row(con, victim["digest"])
                        con.commit()
                        self._unlink_digest(victim["digest"])
                        self.stats.evictions += 1
            except (OSError, sqlite3.Error):
                pass

    def clear(
        self,
        *,
        prefix: str | tuple[Any, ...] | None = None,
        param: str | None = None,
        stage: str | None = None,
    ) -> None:
        if self._disabled:
            return
        with self._lock:
            try:
                with self._connect() as con:
                    rows = con.execute("SELECT * FROM entries").fetchall()
                    for row in rows:
                        if not self._matches(
                            row, prefix=prefix, param=param, stage=stage
                        ):
                            continue
                        self._delete_row(con, row["digest"])
                        self._unlink_digest(row["digest"])
                    con.commit()
            except (OSError, sqlite3.Error):
                pass

    @staticmethod
    def _normalize_prefix(prefix: str | tuple[Any, ...]) -> str:
        if isinstance(prefix, str):
            return prefix.strip("/")
        return "/".join(_key_label(item) for item in prefix if item is not None).strip(
            "/"
        )

    def _matches(
        self,
        row: sqlite3.Row,
        *,
        prefix: str | tuple[Any, ...] | None,
        param: str | None,
        stage: str | None,
    ) -> bool:
        stage_match = (
            stage is None
            or row["stage"] == stage
            or row["stage"].endswith(f".{stage}")
            or row["stage"].endswith(stage)
        )
        if not stage_match:
            return False
        if param is not None:
            labels = self._decode_labels(row["labels"])
            return any(label.startswith(param) for label in labels)
        if prefix is None:
            return True
        normalized = self._normalize_prefix(prefix)
        if not normalized:
            return True
        path = row["path"].strip("/")
        row_stage = row["stage"].strip("/")
        labels = "/".join(self._decode_labels(row["labels"]))
        label_names = self._decode_labels(row["labels"])
        short_path = f"{row_stage.rsplit('.', 1)[-1]}/{labels}".strip("/")
        normalized_path = normalized.replace(".", "/")
        stage_and_param = False
        if "/" in normalized:
            stage_name, param_prefix = normalized.split("/", 1)
            stage_and_param = (
                stage_name in row_stage.split(".")
                and any(
                    label.startswith(param_prefix.split("/", 1)[0])
                    for label in label_names
                )
            )
        return (
            path in (normalized, normalized_path)
            or path.startswith(f"{normalized}/")
            or path.startswith(f"{normalized_path}/")
            or row_stage == normalized
            or row_stage.startswith(f"{normalized}/")
            or row_stage.startswith(f"{normalized}.")
            or row_stage.endswith(f".{normalized}")
            or short_path == normalized
            or short_path.startswith(f"{normalized}/")
            or f"/{normalized}/" in path
            or (
                "/" not in normalized
                and any(label.startswith(normalized) for label in label_names)
            )
            or stage_and_param
        )

    def report(self) -> dict[str, Any]:
        if self._disabled:
            return {"items": 0, "bytes": 0}
        try:
            with self._connect() as con:
                row = con.execute(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(size), 0) AS total "
                    "FROM entries WHERE version = ?",
                    (self.version,),
                ).fetchone()
                return {"items": int(row["n"]), "bytes": int(row["total"])}
        except (OSError, sqlite3.Error):
            return {"items": 0, "bytes": 0}

    def clear_all(self) -> None:
        if self._disabled:
            return
        with self._lock:
            try:
                with self._connect() as con:
                    con.execute("DELETE FROM entries")
                    con.commit()
            except (OSError, sqlite3.Error):
                pass
            try:
                for path in self.data_dir.glob("**/*"):
                    if path.is_file():
                        path.unlink()
            except OSError:
                pass


class CacheManager:
    """Persistent LRU cache with block-level and function-level APIs.

    The manager wraps :class:`joblib.Memory` for backwards compatibility,
    but uses its own bounded, versioned disk index for all new cache
    operations.  Cache access is disabled when ``cache_dir`` is ``None``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        level = int(kwargs.pop("level", 10))
        cache_dir = kwargs.pop("cache_dir", _UNSET)
        version = kwargs.pop("version", kwargs.pop("cache_version", None))
        max_bytes = kwargs.pop(
            "max_bytes",
            kwargs.pop(
                "max_size",
                kwargs.pop("disk_limit", kwargs.pop("max_disk_bytes", None)),
            ),
        )
        max_items = kwargs.pop("max_items", None)
        memory_bytes = kwargs.pop(
            "memory_bytes",
            kwargs.pop(
                "memory_size",
                kwargs.pop("memory_limit", kwargs.pop("max_memory", None)),
            ),
        )
        block_frames = kwargs.pop("block_frames", kwargs.pop("block_size", None))

        if cache_dir is _UNSET:
            cache_dir = args[0] if args else kwargs.get("location")
        elif not args and cache_dir is not None and "location" not in kwargs:
            args = (cache_dir,)

        self.memory: Memory = Memory(*args, **kwargs)
        self.level: int = level
        self._max_bytes = _parse_bytes(max_bytes)
        self._max_items = None if max_items is None else max(0, int(max_items))
        self._memory_bytes = _parse_bytes(memory_bytes)
        self._block_frames = max(1, int(block_frames)) if block_frames else 512
        self._version = self._make_version(version)
        self._cache_dir = (
            None
            if cache_dir is None
            else Path(os.fspath(cache_dir)).expanduser().resolve()
        )
        self._lock = threading.RLock()
        self._memory: OrderedDict[str, tuple[Any, int, str]] = OrderedDict()
        self._memory_size = 0
        self._stats = _Stats()
        self._disk: _DiskBackend | None = None
        self._configure_disk()

    @staticmethod
    def _make_version(version: str | int | None) -> str:
        if version is None:
            return _librosa_version
        return str(version)

    @property
    def _disk_version(self) -> str:
        return f"{self._version}:{_CACHE_FORMAT_VERSION}"

    @property
    def cache_dir(self) -> str | None:
        """Configured cache directory, or ``None`` when disabled."""
        return None if self._cache_dir is None else str(self._cache_dir)

    @property
    def location(self) -> str | None:
        """Alias for :attr:`cache_dir`."""
        return self.cache_dir

    @property
    def version(self) -> str:
        """Cache version tag; changing it invalidates old entries."""
        return self._version

    @property
    def enabled(self) -> bool:
        """Whether persistent caching is active."""
        return self._disk is not None and self._disk.enabled

    @property
    def max_bytes(self) -> int | None:
        """Disk size limit in bytes."""
        return self._max_bytes

    @property
    def disk_limit(self) -> int | None:
        """Alias for :attr:`max_bytes`."""
        return self._max_bytes

    @property
    def max_items(self) -> int | None:
        """Maximum number of disk entries."""
        return self._max_items

    @property
    def memory_bytes(self) -> int | None:
        """In-memory size limit in bytes."""
        return self._memory_bytes

    @property
    def memory_limit(self) -> int | None:
        """Alias for :attr:`memory_bytes`."""
        return self._memory_bytes

    @property
    def block_frames(self) -> int:
        """Number of frames per cached intermediate block."""
        return self._block_frames

    def _configure_disk(self) -> None:
        with self._lock:
            self._memory.clear()
            self._memory_size = 0
            if self._cache_dir is None:
                self._disk = None
                return
            self._disk = _DiskBackend(
                self._cache_dir / _CACHE_ROOT_NAME,
                version=self._disk_version,
                max_bytes=self._max_bytes,
                max_items=self._max_items,
                stats=self._stats,
            )

    def configure(
        self,
        cache_dir: str | os.PathLike[str] | object | None = _UNSET,
        *,
        version: str | int | object | None = _UNSET,
        max_bytes: int | str | object | None = _UNSET,
        max_items: int | object | None = _UNSET,
        memory_bytes: int | str | object | None = _UNSET,
        block_frames: int | object | None = _UNSET,
        level: int | None = None,
    ) -> "CacheManager":
        """Reconfigure the cache at runtime."""
        with self._lock:
            if level is not None:
                self.level = int(level)
            if version is not _UNSET:
                self._version = self._make_version(version)  # type: ignore[arg-type]
            if max_bytes is not _UNSET:
                self._max_bytes = _parse_bytes(max_bytes)  # type: ignore[arg-type]
            if max_items is not _UNSET:
                self._max_items = (
                    None if max_items is None else max(0, int(max_items))  # type: ignore[arg-type]
                )
            if memory_bytes is not _UNSET:
                self._memory_bytes = _parse_bytes(memory_bytes)  # type: ignore[arg-type]
            if block_frames is not _UNSET:
                self._block_frames = max(1, int(block_frames))  # type: ignore[arg-type]
            if cache_dir is not _UNSET:
                self._cache_dir = (
                    None
                    if cache_dir is None
                    else Path(os.fspath(cache_dir)).expanduser().resolve()  # type: ignore[arg-type]
                )
            self._configure_disk()
        return self

    def __call__(
        self, level: int, *, always_wrap: bool = False
    ) -> Callable[[_F], _F]:
        """Cache with an explicitly defined level.

        ``always_wrap`` is used internally by feature functions so that
        programmatic reconfiguration works after import.  The public
        decorator behavior remains unchanged when caching is disabled.
        """

        def wrapper(function: _F) -> _F:
            if not always_wrap and (not self.enabled or self.level < level):
                return function

            @functools.wraps(function)
            def cached_call(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled or self.level < level:
                    return function(*args, **kwargs)
                parent_scope = _DIGEST_SCOPE.get()
                token = _DIGEST_SCOPE.set(
                    {} if parent_scope is None else parent_scope
                )
                try:
                    key, bypass = self._function_key(function, args, kwargs)
                    if bypass:
                        return function(*args, **kwargs)
                    hit, value = self._lookup(key)
                    if hit:
                        return _copy_value(value)
                    value = function(*args, **kwargs)
                    self._store(key, value)
                    return value
                finally:
                    _DIGEST_SCOPE.reset(token)

            return cached_call  # type: ignore[return-value]

        return wrapper

    @staticmethod
    def _bind_arguments(
        function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[tuple[str, Any]], bool]:
        try:
            signature = inspect.signature(function)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return [("args", args), ("kwargs", kwargs)], False

        parameters: list[tuple[str, Any]] = []
        bypass = False
        for name, value in bound.arguments.items():
            param = signature.parameters[name]
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                parameters.extend(
                    (str(key), val) for key, val in sorted(value.items())
                )
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                parameters.extend(
                    (f"{name}[{idx}]", val) for idx, val in enumerate(value)
                )
            else:
                if name == "out" and value is not None:
                    bypass = True
                parameters.append((name, value))
        return parameters, bypass

    def _function_key(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[CacheKey, bool]:
        parameters, bypass = self._bind_arguments(function, args, kwargs)
        stage = (
            f"{getattr(function, '__module__', '')}."
            f"{getattr(function, '__qualname__', function.__name__)}"
        )
        return CacheKey(stage=stage, params=tuple(parameters)), bypass

    def key(self, stage: str, /, **params: Any) -> CacheKey:
        """Construct a named cache key."""
        return cache_key(stage, **params)

    @staticmethod
    def _coerce_key(
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any]
    ) -> tuple[str, tuple[tuple[str, Any], ...]]:
        if isinstance(key, CacheKey):
            return key.stage, key.params
        if isinstance(key, str):
            return key, ()
        if isinstance(key, dict):
            stage = str(key.get("stage", "cache"))
            return stage, tuple(
                (str(name), value) for name, value in key.items() if name != "stage"
            )
        if isinstance(key, tuple):
            if not key:
                return "cache", ()
            stage = str(key[0])
            rest = key[1:]
            if rest and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in rest
            ):
                return stage, tuple((item[0], item[1]) for item in rest)  # type: ignore[arg-type]
            return stage, tuple(
                (f"arg{idx}", value) for idx, value in enumerate(rest)
            )
        return type(key).__name__, (("value", key),)

    @staticmethod
    def _path(stage: str, params: tuple[tuple[str, Any], ...]) -> str:
        safe_stage = "/".join(_key_label(part) for part in stage.split("."))
        labels = "/".join(_key_label(name) for name, _ in params)
        return f"{safe_stage}/{labels}".rstrip("/")

    def prepare(
        self, key: CacheKey | str | tuple[Any, ...] | dict[str, Any]
    ) -> _PreparedKey:
        """Precompute the expensive part of a cache key once."""
        stage, params = self._coerce_key(key)
        digest = hashlib.blake2b(digest_size=32)
        digest.update(b"librosa-cache")
        digest.update(str(_CACHE_FORMAT_VERSION).encode("ascii"))
        digest.update(self._version.encode("utf-8", "surrogatepass"))
        digest.update(stage.encode("utf-8", "surrogatepass"))
        digest.update(struct.pack("<q", len(params)))
        for name, value in params:
            digest.update(name.encode("utf-8", "surrogatepass"))
            digest.update(_digest_value(value))
        return _PreparedKey(
            stage=stage,
            params=params,
            digest=digest.digest(),
            path=self._path(stage, params),
        )

    @staticmethod
    def _block_digest(prepared: _PreparedKey, block: int | str) -> str:
        digest = hashlib.blake2b(digest_size=32)
        digest.update(prepared.digest)
        digest.update(repr(block).encode("utf-8", "surrogatepass"))
        return digest.hexdigest()

    def _digest(
        self,
        stage: str,
        params: tuple[tuple[str, Any], ...],
        block: int | str,
    ) -> str:
        return self._block_digest(self.prepare(CacheKey(stage, params)), block)

    def _memory_put(self, digest: str, value: Any, stage: str) -> None:
        if self._memory_bytes is not None and self._memory_bytes <= 0:
            return
        size = _nbytes(value)
        if self._memory_bytes is not None and size > self._memory_bytes:
            self._stats.skipped += 1
            return
        with self._lock:
            old = self._memory.pop(digest, None)
            if old is not None:
                self._memory_size -= old[1]
            while self._memory and (
                self._memory_bytes is not None
                and self._memory_size + size > self._memory_bytes
            ):
                _, (_, old_size, _) = self._memory.popitem(last=False)
                self._memory_size -= old_size
                self._stats.evictions += 1
            self._memory[digest] = (value, size, stage)
            self._memory_size += size

    def _memory_get(self, digest: str) -> tuple[bool, Any]:
        with self._lock:
            entry = self._memory.get(digest)
            if entry is None:
                return False, None
            self._memory.move_to_end(digest)
            return True, entry[0]

    def _lookup_prepared(
        self, prepared: _PreparedKey, block: int | str = 0
    ) -> tuple[bool, Any]:
        if not self.enabled:
            return False, None
        stage = prepared.stage
        digest = self._block_digest(prepared, block)
        hit, value = self._memory_get(digest)
        if hit:
            if self._disk is not None:
                self._disk.touch(digest)
            self._stats.hit(stage, "memory")
            return True, value
        if self._disk is None:
            self._stats.miss(stage)
            return False, None
        hit, value = self._disk.lookup(digest)
        if hit:
            self._memory_put(digest, value, stage)
            self._stats.hit(stage, "disk")
            return True, value
        self._stats.miss(stage)
        return False, None

    def _lookup(
        self, key: CacheKey | str | tuple[Any, ...] | dict[str, Any], block: int | str = 0
    ) -> tuple[bool, Any]:
        return self._lookup_prepared(self.prepare(key), block=block)

    def _store_prepared(
        self, prepared: _PreparedKey, value: Any, block: int | str = 0
    ) -> None:
        if not self.enabled:
            return
        stage = prepared.stage
        digest = self._block_digest(prepared, block)
        if self._disk is not None:
            self._disk.put(
                digest,
                value=value,
                stage=stage,
                labels=[name for name, _ in prepared.params],
            )
            self._stats.writes += 1
        self._memory_put(digest, value, stage)

    def _store(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        value: Any,
        block: int | str = 0,
    ) -> None:
        self._store_prepared(self.prepare(key), value, block=block)

    def lookup_prepared(
        self, prepared: _PreparedKey, block: int | str = 0
    ) -> tuple[bool, Any]:
        """Return ``(hit, value)`` for a prepared key."""
        return self._lookup_prepared(prepared, block=block)

    def put_prepared(
        self, prepared: _PreparedKey, value: Any, block: int | str = 0
    ) -> None:
        """Store a value for a prepared key."""
        self._store_prepared(prepared, value, block=block)

    def lookup(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        block: int | str = 0,
    ) -> tuple[bool, Any]:
        """Return ``(hit, value)`` for a key and optional block."""
        return self._lookup(key, block=block)

    def get(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        block: int | str = 0,
        default: Any = None,
    ) -> Any:
        """Return a cached value or ``default`` on a miss."""
        hit, value = self._lookup(key, block=block)
        if not hit:
            return default
        return _copy_value(value)

    def put(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        value: Any,
        block: int | str = 0,
    ) -> None:
        """Store a value for a key and optional block."""
        self._store(key, value, block=block)

    def contains(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        block: int | str = 0,
    ) -> bool:
        """Return whether a key is currently cached."""
        hit, _ = self._lookup(key, block=block)
        return hit

    def get_or_compute(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        compute: Callable[[], Any],
        block: int | str = 0,
    ) -> Any:
        """Return a cached value, or compute and store it."""
        hit, value = self._lookup(key, block=block)
        if hit:
            return _copy_value(value)
        value = compute()
        self._store(key, value, block=block)
        return value

    def delete(
        self,
        key: CacheKey | str | tuple[Any, ...] | dict[str, Any],
        block: int | str = 0,
    ) -> None:
        """Remove one cache entry."""
        stage, params = self._coerce_key(key)
        digest = self._digest(stage, params, block)
        with self._lock:
            self._memory.pop(digest, None)
        if self._disk is not None:
            self._disk.delete(digest)

    def clear(
        self,
        prefix: str | tuple[Any, ...] | None = None,
        *,
        param: str | None = None,
        stage: str | None = None,
    ) -> None:
        """Clear cache entries."""
        with self._lock:
            self._memory.clear()
            self._memory_size = 0
            if self._disk is not None:
                if prefix is None and param is None and stage is None:
                    self._disk.clear_all()
                else:
                    self._disk.clear(prefix=prefix, param=param, stage=stage)
            if prefix is None and param is None and stage is None:
                with contextlib.suppress(Exception):
                    self.memory.clear()

    def clear_prefix(self, prefix: str | tuple[Any, ...]) -> None:
        """Invalidate all entries under a stage or parameter prefix."""
        self.clear(prefix=prefix)

    def invalidate_prefix(self, prefix: str | tuple[Any, ...]) -> None:
        """Alias for :meth:`clear_prefix`."""
        self.clear_prefix(prefix)

    def clear_param(self, param: str, *, stage: str | None = None) -> None:
        """Invalidate all entries containing a parameter name prefix."""
        self.clear(param=param, stage=stage)

    def invalidate_param(self, param: str, *, stage: str | None = None) -> None:
        """Alias for :meth:`clear_param`."""
        self.clear_param(param, stage=stage)

    def invalidate(
        self,
        *,
        prefix: str | tuple[Any, ...] | None = None,
        param: str | None = None,
        stage: str | None = None,
    ) -> None:
        """Alias for :meth:`clear` with keyword filtering."""
        self.clear(prefix=prefix, param=param, stage=stage)

    def reset_stats(self) -> None:
        """Reset hit/miss counters without deleting cached values."""
        with self._lock:
            self._stats = _Stats()
            if self._disk is not None:
                self._disk.stats = self._stats

    def report(self) -> CacheStats:
        """Return a snapshot of cache usage and hit statistics."""
        disk = (
            self._disk.report() if self._disk is not None else {"items": 0, "bytes": 0}
        )
        with self._lock:
            memory_items = len(self._memory)
            memory_bytes = self._memory_size
            stats = self._stats
            report = {
                "enabled": self.enabled,
                "cache_dir": self.cache_dir,
                "version": self.version,
                "requests": stats.requests,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hits / stats.requests if stats.requests else 0.0,
                "memory_hits": stats.memory_hits,
                "disk_hits": stats.disk_hits,
                "writes": stats.writes,
                "evictions": stats.evictions,
                "corruptions": stats.corruptions,
                "skipped": stats.skipped,
                "memory_items": memory_items,
                "memory_bytes": memory_bytes,
                "memory_limit": self.memory_bytes,
                "disk_items": disk["items"],
                "disk_bytes": disk["bytes"],
                "disk_limit": self.max_bytes,
                "block_frames": self.block_frames,
                "by_stage": {
                    stage: dict(values) for stage, values in stats.by_stage.items()
                },
            }
        return CacheStats(report)

    def report_stats(self) -> CacheStats:
        """Alias for :meth:`report`."""
        return self.report()

    def get_stats(self) -> CacheStats:
        """Alias for :meth:`report`."""
        return self.report()

    @property
    def stats(self) -> CacheStats:
        """Current cache statistics."""
        return self.report()

    @property
    def hit_rate(self) -> float:
        """Fraction of cache lookups that were hits."""
        stats = self._stats
        return stats.hits / stats.requests if stats.requests else 0.0

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate a function through the legacy joblib backend."""
        return self.memory.eval(*args, **kwargs)

    def format(self, *args: Any, **kwargs: Any) -> Any:
        """Return the formatted representation of an object."""
        return self.memory.format(*args, **kwargs)

    def reduce_size(self, *args: Any, **kwargs: Any) -> None:
        """Enforce configured limits and optionally call joblib's reducer."""
        if self._disk is not None:
            self._disk._evict()
        self._trim_memory()
        with contextlib.suppress(Exception):
            self.memory.reduce_size(*args, **kwargs)

    def _trim_memory(self) -> None:
        if self._memory_bytes is None:
            return
        with self._lock:
            while self._memory and self._memory_size > self._memory_bytes:
                _, (_, size, _) = self._memory.popitem(last=False)
                self._memory_size -= size
                self._stats.evictions += 1

    def warn(self, *args: Any, **kwargs: Any) -> None:
        """Raise a warning through the legacy joblib backend."""
        self.memory.warn(*args, **kwargs)  # pragma: no cover


# Keep the public class name available for callers that want block semantics.
BlockCache = CacheManager


def _env_bytes(names: Iterable[str], default: int | None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return _parse_bytes(value, default=default)
    return default


def _env_int(names: Iterable[str], default: int | None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


# Instantiate the cache from the environment.
cache: CacheManager = CacheManager(
    os.environ.get("LIBROSA_CACHE_DIR", None),
    mmap_mode=os.environ.get("LIBROSA_CACHE_MMAP", None),
    compress=_parse_bool(os.environ.get("LIBROSA_CACHE_COMPRESS", False)),
    verbose=int(os.environ.get("LIBROSA_CACHE_VERBOSE", 0)),
    level=int(os.environ.get("LIBROSA_CACHE_LEVEL", 10)),
    max_bytes=_env_bytes(
        ("LIBROSA_CACHE_MAX_BYTES", "LIBROSA_CACHE_MAX_DISK_BYTES"),
        512 * 1024**2,
    ),
    max_items=_env_int(("LIBROSA_CACHE_MAX_ITEMS",), 4096),
    memory_bytes=_env_bytes(
        ("LIBROSA_CACHE_MEMORY_BYTES", "LIBROSA_CACHE_MAX_MEMORY_BYTES"),
        128 * 1024**2,
    ),
    block_frames=_env_int(("LIBROSA_CACHE_BLOCK_FRAMES",), 512),
    version=os.environ.get("LIBROSA_CACHE_VERSION", None),
)
