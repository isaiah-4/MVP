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

        wrapped_model = ThreadSafeYOLOModel(YOLO(str(resolved_path)))
        _MODEL_CACHE[cache_key] = wrapped_model
        return wrapped_model


class ThreadSafeYOLOModel:
    def __init__(self, model):
        self._model = model
        self._predict_lock = Lock()

    def predict(self, *args, **kwargs):
        # Ultralytics model instances are not thread-safe for concurrent inference.
        with self._predict_lock:
            return self._model.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _build_cache_key(model_path):
    stat_result = model_path.stat()
    return (
        str(model_path),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )
