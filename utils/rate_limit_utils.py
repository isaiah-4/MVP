from __future__ import annotations

from collections import defaultdict, deque
from functools import wraps
import threading
from time import monotonic

from fastapi import Request
from fastapi.responses import JSONResponse


try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

    class RateLimitExceeded(Exception):
        def __init__(self, detail: str = "Rate limit exceeded"):
            super().__init__(detail)
            self.detail = detail


    def _rate_limit_exceeded_handler(request: Request, exc: Exception):
        detail = getattr(exc, "detail", "Rate limit exceeded")
        return JSONResponse({"detail": str(detail)}, status_code=429)


    def get_remote_address(request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        client = getattr(request, "client", None)
        if client is not None and getattr(client, "host", None):
            return str(client.host)
        return "unknown"


    class Limiter:
        def __init__(self, *, key_func, default_limits=None):
            self.key_func = key_func
            self.default_limits = list(default_limits or [])
            self._lock = threading.Lock()
            self._buckets = defaultdict(deque)

        def limit(self, limit_value: str):
            max_requests, window_seconds = _parse_limit(limit_value)

            def decorator(func):
                route_identifier = f"{func.__module__}.{func.__qualname__}"

                @wraps(func)
                async def wrapper(*args, **kwargs):
                    request = _extract_request(args, kwargs)
                    if request is None:
                        raise RuntimeError(
                            "Rate-limited routes must accept a FastAPI Request argument."
                        )

                    bucket_key = (route_identifier, self.key_func(request))
                    now = monotonic()

                    with self._lock:
                        bucket = self._buckets[bucket_key]
                        _prune_bucket(bucket, now=now, window_seconds=window_seconds)
                        if len(bucket) >= max_requests:
                            raise RateLimitExceeded()
                        bucket.append(now)

                    return await func(*args, **kwargs)

                return wrapper

            return decorator


    def _parse_limit(limit_value: str) -> tuple[int, float]:
        raw_value = str(limit_value or "").strip()
        if "/" not in raw_value:
            raise ValueError(f"Unsupported rate limit format: {limit_value!r}")

        count_part, window_part = [segment.strip().lower() for segment in raw_value.split("/", 1)]
        max_requests = int(count_part)
        if max_requests <= 0:
            raise ValueError("Rate limit request count must be positive.")

        if window_part in {"second", "sec", "s"}:
            return max_requests, 1.0
        if window_part in {"minute", "min", "m"}:
            return max_requests, 60.0
        if window_part in {"hour", "hr", "h"}:
            return max_requests, 3600.0
        if window_part in {"day", "d"}:
            return max_requests, 86400.0

        raise ValueError(f"Unsupported rate limit window: {limit_value!r}")


    def _extract_request(args, kwargs) -> Request | None:
        request = kwargs.get("request")
        if isinstance(request, Request):
            return request

        for value in args:
            if isinstance(value, Request):
                return value

        return None


    def _prune_bucket(bucket, *, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
