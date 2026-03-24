from pathlib import Path
from threading import Lock

from ultralytics import YOLO


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = Lock()


def get_yolo_model(model_path):
    resolved_path = Path(model_path).expanduser().resolve()
    cache_key = _build_cache_key(resolved_path)

    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        model = YOLO(str(resolved_path))
        _MODEL_CACHE[cache_key] = model
        return model


def _build_cache_key(model_path):
    stat_result = model_path.stat()
    return (
        str(model_path),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )
