import os
from pathlib import Path
from threading import Lock

import torch
from ultralytics import YOLO


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = Lock()
_DEVICE = None


def get_inference_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    env_device = os.environ.get("COURTVISION_INFERENCE_DEVICE")
    if env_device:
        _DEVICE = env_device
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = "cuda"
        return _DEVICE

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        _DEVICE = "mps"
        return _DEVICE

    _DEVICE = "cpu"
    return _DEVICE


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
        self._device = get_inference_device()

    def predict(self, *args, **kwargs):
        kwargs.setdefault("device", self._device)
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
