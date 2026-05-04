from __future__ import annotations

import gzip
import logging
from pathlib import Path

import msgpack
import numpy as np


LOGGER = logging.getLogger(__name__)
SERIALIZATION_MARKER = "__courtvision_type__"
LEGACY_PICKLE_MAGIC = b"\x80"
LEGACY_PICKLE_WARNED_PATHS: set[Path] = set()


def save_stub(stub_path, obj, *, compress: bool = True):
    if stub_path is None:
        return

    path = Path(stub_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = msgpack.packb(
        obj,
        default=_pack_msgpack_extension,
        use_bin_type=True,
    )

    if compress:
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
        return

    path.write_bytes(payload)


def read_stub(read_from_stub, stub_path):
    if not read_from_stub or stub_path is None:
        return None

    path = Path(stub_path)
    if not path.exists():
        legacy_path = _build_legacy_pickle_path(path)
        if legacy_path is not None and legacy_path.exists():
            _warn_legacy_pickle(legacy_path, path)
        return None

    try:
        payload = path.read_bytes()
        if _looks_like_pickle(payload):
            _warn_legacy_pickle(path, path)
            return None
        if _looks_like_gzip(payload):
            payload = gzip.decompress(payload)

        return msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=False,
            object_hook=_unpack_msgpack_extension,
        )
    except Exception:
        LOGGER.warning(
            "Failed to read stub %s; it will be regenerated.",
            path.name,
            exc_info=True,
        )
        return None


def _pack_msgpack_extension(value):
    if isinstance(value, np.ndarray):
        return {
            SERIALIZATION_MARKER: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return {
            SERIALIZATION_MARKER: "path",
            "value": str(value),
        }
    if isinstance(value, set):
        return {
            SERIALIZATION_MARKER: "set",
            "items": list(value),
        }
    raise TypeError(f"Unsupported stub value type: {type(value)!r}")


def _unpack_msgpack_extension(value):
    marker = value.get(SERIALIZATION_MARKER)
    if marker == "ndarray":
        return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
    if marker == "path":
        return Path(value["value"])
    if marker == "set":
        return set(value["items"])
    return value


def _build_legacy_pickle_path(path: Path) -> Path | None:
    if path.suffix == ".pkl":
        return path
    if path.name.endswith(".msgpack.gz"):
        return path.with_name(f"{path.name[:-len('.msgpack.gz')]}.pkl")
    if path.suffix == ".msgpack":
        return path.with_suffix(".pkl")
    return None


def _looks_like_gzip(payload: bytes) -> bool:
    return payload[:2] == b"\x1f\x8b"


def _looks_like_pickle(payload: bytes) -> bool:
    return len(payload) >= 2 and payload[:1] == LEGACY_PICKLE_MAGIC and payload[1] <= 5


def _warn_legacy_pickle(legacy_path: Path, target_path: Path) -> None:
    if legacy_path in LEGACY_PICKLE_WARNED_PATHS:
        return

    LEGACY_PICKLE_WARNED_PATHS.add(legacy_path)
    LOGGER.warning(
        "Legacy pickle stub %s detected. It will be ignored and regenerated as %s on this run.",
        legacy_path.name,
        target_path.name,
    )
