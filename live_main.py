import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from analysis_pipeline import (
    AnalysisRunResult,
    RESULT_CACHE_VERSION,
    run_analysis,
    run_chunked_full_analysis,
)
from utils.input_utils import (
    InputSourceValidationError,
    build_source_key,
    extract_youtube_id,
    is_youtube_url,
    validate_input_source as validate_requested_input_source,
)
from utils.rate_limit_utils import (
    Limiter,
    RateLimitExceeded,
    SLOWAPI_AVAILABLE,
    _rate_limit_exceeded_handler,
    get_remote_address,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))
DEFAULT_MAX_WORKERS = max(1, int(os.environ.get("COURTVISION_MAX_WORKERS", "1")))
JOB_TTL_SECONDS = int(os.environ.get("COURTVISION_JOB_TTL_SECONDS", str(6 * 60 * 60)))
RESULT_CACHE_TTL_SECONDS = int(
    os.environ.get("COURTVISION_RESULT_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)
MAX_COMPLETED_JOBS = int(os.environ.get("COURTVISION_MAX_COMPLETED_JOBS", "100"))
MAX_CACHED_RESULTS = int(os.environ.get("COURTVISION_MAX_CACHED_RESULTS", "64"))
DISK_CACHE_KEY_PREFIX = "disk-cache:"
RATE_LIMIT = os.environ.get("COURTVISION_RATE_LIMIT", "30/minute").strip() or "30/minute"
WORKOUT_ACCOUNTS = (
    {"id": "primary-athlete", "label": "Primary Athlete"},
    {"id": "development-guard", "label": "Development Guard"},
    {"id": "varsity-wing", "label": "Varsity Wing"},
)
ANALYSIS_PROFILES = {
    "preview": {
        "label": "Fast Preview",
        "run_suffix": "preview_300",
        "max_dimension": 720,
        "max_frames": 300,
        "court_keypoint_interval": 8,
    },
    "full": {
        "label": "Full Run",
        "run_suffix": "full",
        "max_frames": None,
        "chunk_frames": 300,
        "court_keypoint_interval": 12,
    },
}
EXECUTOR = ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS)
JOB_LOCK = asyncio.Lock()
RESULT_CACHE = {}
RESULT_CACHE_UPDATED_AT = {}
ACTIVE_JOBS_BY_KEY = {}
JOBS = {}


@dataclass
class AnalysisJob:
    job_id: str
    cache_key: str
    input_source: str
    session_name: str
    mode: str
    player_id: str
    account_id: str
    analysis_profile: str
    status: str = "queued"
    error: str | None = None
    result: AnalysisRunResult | None = None
    progress: float = 0.0
    progress_message: str = "Queued"
    processed_frames: int = 0
    total_frames: int = 0
    processed_chunks: int = 0
    total_chunks: int = 0
    partial_outputs: list[dict] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


def get_api_token() -> str:
    return os.environ.get("COURTVISION_API_TOKEN", "").strip()


def resolve_cors_origins() -> list[str]:
    configured_origins = os.environ.get("COURTVISION_CORS_ORIGINS", "*").strip()
    if not configured_origins:
        return []
    if configured_origins == "*":
        return ["*"]
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


app = FastAPI(title="CourtVision MVP", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="assets")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "Output_vids")), name="outputs")
limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_home_context(request, **overrides):
    context = {
        "request": request,
        "error": None,
        "form": {
            "session_name": "Game ...",
            "mode": "game",
            "analysis_profile": "preview",
            "player_id": "",
            "account_id": "",
            "input": "Input_vids/video_1.mp4",
        },
        "workout_accounts": get_workout_accounts(),
        "available_now": [
            "Processed feed with player, ball, court, and tactical overlays",
            "Team-scoped player labels with appearance-based identity cleanup and optional TrOCR jersey OCR",
            "Touch counts from confirmed ball possession changes",
            "Average possession time per tracked player",
            "Team pass and interception counts",
            "First-pass shot attempts and makes/misses from ball, hoop, and release heuristics",
        ],
        "coming_next": [
            "Improvements to the shot and ball tracking as well as the team assigning and deflection tracking",
        ],
    }
    context.update(overrides)
    return context


def build_prototype_context(request, **overrides):
    bootstrap = {
        "error": None,
        "form": {
            "session_name": "",
            "mode": "game",
            "analysis_profile": "preview",
            "player_id": "",
            "account_id": "",
            "input": "",
        },
        "workout_accounts": get_workout_accounts(),
        "analysis_profiles": [
            {"id": profile_id, "label": profile["label"]}
            for profile_id, profile in ANALYSIS_PROFILES.items()
        ],
    }
    override_bootstrap = overrides.pop("bootstrap", None)
    if override_bootstrap:
        bootstrap.update(override_bootstrap)
    return {
        "request": request,
        "bootstrap": bootstrap,
        **overrides,
    }


def get_analysis_profile_config(analysis_profile):
    if analysis_profile not in ANALYSIS_PROFILES:
        raise ValueError(f"Unknown analysis profile: {analysis_profile}")
    return ANALYSIS_PROFILES[analysis_profile]


def get_workout_accounts():
    return [dict(account) for account in WORKOUT_ACCOUNTS]


def get_workout_account_label(account_id):
    normalized_account_id = str(account_id or "").strip()
    for account in WORKOUT_ACCOUNTS:
        if account["id"] == normalized_account_id:
            return account["label"]
    return ""


def validate_session_options(mode, player_id, account_id):
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"game", "workout"}:
        raise ValueError(f"Unknown session mode: {mode}")

    if normalized_mode != "workout":
        return

    if not str(account_id or "").strip():
        raise ValueError("Workout mode requires an account selection.")

    if not get_workout_account_label(account_id):
        raise ValueError("Unknown workout account selected.")

    if not str(player_id or "").strip():
        raise ValueError("Workout mode requires a player number.")


def normalize_input_key(input_source, analysis_profile, mode, player_id, account_id):
    source = input_source.strip()
    profile_key = (
        "profile:"
        f"{analysis_profile}:mode:{mode}:player:{player_id.strip()}:"
        f"account:{str(account_id or '').strip()}:analysis:{RESULT_CACHE_VERSION}"
    )
    if is_youtube_url(source):
        video_id = extract_youtube_id(source)
        if video_id is not None:
            return f"{profile_key}:yt:{video_id}"
        return f"{profile_key}:yt:{source}"

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if source_path.exists():
        return f"{profile_key}:file:{source_path}:{source_path.stat().st_mtime_ns}"

    return f"{profile_key}:file:{source_path}"


def build_base_source_key(input_source):
    source = input_source.strip()
    if is_youtube_url(source):
        return build_source_key(source)

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    return build_source_key(source, resolved_path=source_path)


def result_matches_request(result, input_source, analysis_profile, mode, player_id):
    profile_config = get_analysis_profile_config(analysis_profile)
    expected_base_source_key = build_base_source_key(input_source)

    return (
        str(result.base_source_key) == str(expected_base_source_key)
        and str(result.mode) == str(mode)
        and str(result.workout_player_id or "") == str(player_id or "")
        and result.max_frames == profile_config.get("max_frames")
        and result.chunk_frames == profile_config.get("chunk_frames")
        and result.max_dimension == profile_config.get("max_dimension")
        and int(result.court_keypoint_interval)
        == int(profile_config.get("court_keypoint_interval", 12))
    )


def find_semantic_cached_result(input_source, analysis_profile, mode, player_id):
    seen_output_paths = set()
    for cache_key, cached_result in RESULT_CACHE.items():
        if cached_result is None:
            continue
        output_path = str(cached_result.output_path)
        if output_path in seen_output_paths:
            continue
        seen_output_paths.add(output_path)
        if result_matches_request(
            cached_result,
            input_source,
            analysis_profile,
            mode,
            player_id,
        ):
            return cache_key, cached_result
    return None, None


def _scan_disk_cached_results():
    output_dir = PROJECT_ROOT / "Output_vids"
    if not output_dir.exists():
        return []

    entries = []
    for cache_path in output_dir.glob("*.json"):
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        if int(payload.get("cache_version", 0)) != RESULT_CACHE_VERSION:
            continue

        try:
            cached_result = AnalysisRunResult.from_public_dict(payload)
        except Exception:
            continue

        if not cached_result.output_path.exists():
            continue

        entries.append(
            (
                f"{DISK_CACHE_KEY_PREFIX}{cache_path.name}",
                cached_result,
                float(cache_path.stat().st_mtime),
            )
        )

    return entries


async def warm_result_cache_from_disk():
    entries = await run_blocking(_scan_disk_cached_results)
    async with JOB_LOCK:
        for cache_key, cached_result, updated_at in entries:
            RESULT_CACHE[cache_key] = cached_result
            RESULT_CACHE_UPDATED_AT[cache_key] = updated_at
        prune_state_locked()


def render_dashboard(
    request,
    result,
    session_name,
    mode,
    player_id,
    account_id,
    input_source,
    analysis_profile,
):
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "session_name": session_name,
            "mode": mode,
            "analysis_profile": analysis_profile,
            "analysis_profile_label": ANALYSIS_PROFILES[analysis_profile]["label"],
            "player_id": player_id.strip(),
            "account_id": account_id.strip(),
            "account_name": get_workout_account_label(account_id),
            "input_source": input_source,
            "input_path": str(result.input_path),
            "source_key": result.source_key,
            "output_url": f"/outputs/{result.output_path.name}",
            "using_court_model": result.using_court_model,
            "max_frames": result.max_frames,
            "processed_frames": result.processed_frames,
            "chunk_count": result.chunk_count or result.session_metrics.get("overview", {}).get("chunk_count", 1),
            "player_ids_are_chunk_local": result.player_ids_are_chunk_local,
            "chunk_outputs": list(result.chunk_outputs),
            "metrics": result.session_metrics,
            "cached_result": True,
        },
    )


async def submit_or_get_job(input_source, session_name, mode, player_id, account_id, analysis_profile):
    get_analysis_profile_config(analysis_profile)
    validate_session_options(mode, player_id, account_id)
    normalized_input_source = await run_blocking(validate_requested_input_source, input_source)
    cache_key = normalize_input_key(
        normalized_input_source,
        analysis_profile,
        mode,
        player_id,
        account_id,
    )

    async with JOB_LOCK:
        prune_state_locked()
        cached_result = RESULT_CACHE.get(cache_key)
        if cached_result is not None:
            return normalized_input_source, cached_result, None

        matched_cache_key, cached_result = find_semantic_cached_result(
            normalized_input_source,
            analysis_profile,
            mode,
            player_id,
        )
        if cached_result is not None:
            RESULT_CACHE[cache_key] = cached_result
            RESULT_CACHE_UPDATED_AT[cache_key] = RESULT_CACHE_UPDATED_AT.get(
                matched_cache_key,
                time(),
            )
            return normalized_input_source, cached_result, None

        active_job_id = ACTIVE_JOBS_BY_KEY.get(cache_key)
        if active_job_id is not None:
            return normalized_input_source, None, JOBS[active_job_id]

        job_id = uuid4().hex
        job = AnalysisJob(
            job_id=job_id,
            cache_key=cache_key,
            input_source=normalized_input_source,
            session_name=session_name,
            mode=mode,
            player_id=player_id,
            account_id=account_id,
            analysis_profile=analysis_profile,
            created_at=time(),
            updated_at=time(),
        )
        JOBS[job_id] = job
        ACTIVE_JOBS_BY_KEY[cache_key] = job_id

    asyncio.create_task(run_job(job_id))
    return normalized_input_source, None, job


async def run_job(job_id):
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.progress_message = "Starting analysis"
        job.updated_at = time()

    try:
        result = await execute_job_analysis(job)
    except Exception as exc:
        LOGGER.exception("Analysis job %s failed", job_id)
        async with JOB_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = build_safe_job_error(exc)
            job.updated_at = time()
            ACTIVE_JOBS_BY_KEY.pop(job.cache_key, None)
        return

    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.status = "completed"
        job.result = result
        job.progress = 1.0
        job.progress_message = "Completed"
        job.processed_frames = result.processed_frames
        job.total_frames = result.processed_frames
        job.processed_chunks = result.chunk_count or 1
        job.total_chunks = result.chunk_count or 1
        job.partial_outputs = list(result.chunk_outputs)
        job.updated_at = time()
        RESULT_CACHE[job.cache_key] = result
        RESULT_CACHE_UPDATED_AT[job.cache_key] = job.updated_at
        ACTIVE_JOBS_BY_KEY.pop(job.cache_key, None)
        prune_state_locked()


async def execute_job_analysis(job):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        EXECUTOR,
        lambda: run_job_sync(job, loop),
    )


def run_job_sync(job, loop):
    profile_config = get_analysis_profile_config(job.analysis_profile)
    progress_callback = lambda payload: dispatch_job_progress(loop, job.job_id, payload)

    if job.analysis_profile == "full":
        return run_chunked_full_analysis(
            job.input_source,
            run_suffix=profile_config["run_suffix"],
            chunk_frames=profile_config.get("chunk_frames", 300),
            max_dimension=profile_config.get("max_dimension"),
            court_keypoint_interval=profile_config.get("court_keypoint_interval", 12),
            mode=job.mode,
            workout_player_id=job.player_id,
            progress_callback=progress_callback,
        )

    dispatch_job_progress(
        loop,
        job.job_id,
        {
            "progress": 0.0,
            "progress_message": "Analyzing preview",
        },
    )
    return run_analysis(
        job.input_source,
        max_dimension=profile_config.get("max_dimension"),
        max_frames=profile_config["max_frames"],
        run_suffix=profile_config["run_suffix"],
        court_keypoint_interval=profile_config.get("court_keypoint_interval", 12),
        mode=job.mode,
        workout_player_id=job.player_id,
        progress_callback=progress_callback,
    )


def dispatch_job_progress(loop, job_id, payload):
    future = asyncio.run_coroutine_threadsafe(update_job_progress(job_id, payload), loop)
    future.add_done_callback(log_progress_callback_failure)


def log_progress_callback_failure(future: Future):
    try:
        error = future.exception()
    except Exception as exc:
        LOGGER.warning(
            "Failed to read progress callback result",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return

    if error is not None:
        LOGGER.warning(
            "Failed to update job progress",
            exc_info=(type(error), error, error.__traceback__),
        )


async def update_job_progress(job_id, payload):
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.status == "failed":
            return

        if "progress" in payload:
            job.progress = float(payload["progress"])
        if "progress_message" in payload:
            job.progress_message = str(payload["progress_message"])
        if "processed_frames" in payload:
            job.processed_frames = int(payload["processed_frames"])
        if "total_frames" in payload:
            job.total_frames = int(payload["total_frames"])
        if "processed_chunks" in payload:
            job.processed_chunks = int(payload["processed_chunks"])
        if "total_chunks" in payload:
            job.total_chunks = int(payload["total_chunks"])
        if "partial_outputs" in payload:
            job.partial_outputs = list(payload["partial_outputs"])
        job.updated_at = time()


async def get_job(job_id):
    async with JOB_LOCK:
        prune_state_locked()
        return JOBS.get(job_id)


def prune_state_locked(now=None):
    now = time() if now is None else float(now)

    expired_cache_keys = [
        cache_key
        for cache_key, updated_at in RESULT_CACHE_UPDATED_AT.items()
        if (now - updated_at) > RESULT_CACHE_TTL_SECONDS
    ]
    for cache_key in expired_cache_keys:
        RESULT_CACHE.pop(cache_key, None)
        RESULT_CACHE_UPDATED_AT.pop(cache_key, None)

    if len(RESULT_CACHE_UPDATED_AT) > MAX_CACHED_RESULTS:
        overflow_count = len(RESULT_CACHE_UPDATED_AT) - MAX_CACHED_RESULTS
        oldest_cache_keys = sorted(
            RESULT_CACHE_UPDATED_AT,
            key=RESULT_CACHE_UPDATED_AT.get,
        )[:overflow_count]
        for cache_key in oldest_cache_keys:
            RESULT_CACHE.pop(cache_key, None)
            RESULT_CACHE_UPDATED_AT.pop(cache_key, None)

    completed_jobs = [
        job
        for job in JOBS.values()
        if job.status in {"completed", "failed"}
    ]
    expired_job_ids = {
        job.job_id
        for job in completed_jobs
        if (now - job.updated_at) > JOB_TTL_SECONDS
    }

    remaining_completed_jobs = [
        job
        for job in completed_jobs
        if job.job_id not in expired_job_ids
    ]
    if len(remaining_completed_jobs) > MAX_COMPLETED_JOBS:
        overflow_count = len(remaining_completed_jobs) - MAX_COMPLETED_JOBS
        oldest_jobs = sorted(
            remaining_completed_jobs,
            key=lambda job: job.updated_at,
        )[:overflow_count]
        expired_job_ids.update(job.job_id for job in oldest_jobs)

    for job_id in expired_job_ids:
        job = JOBS.pop(job_id, None)
        if job is None:
            continue
        if ACTIVE_JOBS_BY_KEY.get(job.cache_key) == job_id:
            ACTIVE_JOBS_BY_KEY.pop(job.cache_key, None)


async def run_blocking(func, *args, executor=None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: func(*args))


def build_safe_request_error(exc):
    if isinstance(exc, (InputSourceValidationError, ValueError)):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "Input video was not found."
    return "Unable to start analysis."


def build_safe_job_error(exc):
    if isinstance(exc, InputSourceValidationError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "Input video was not found."
    if isinstance(exc, ValueError) and str(exc).startswith("Could not open video file:"):
        return "Could not open the input video."
    return str(exc) or "Analysis failed."


def extract_request_token(request: Request) -> str:
    authorization_header = request.headers.get("Authorization", "").strip()
    if authorization_header.startswith("Bearer "):
        return authorization_header[7:].strip()
    return request.headers.get("X-API-Token", "").strip()


@app.middleware("http")
async def enforce_api_token(request: Request, call_next):
    api_token = get_api_token()
    if not api_token or request.method.upper() == "OPTIONS" or request.url.path == "/health":
        return await call_next(request)

    provided_token = extract_request_token(request)
    if not provided_token or provided_token != api_token:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return await call_next(request)


@app.on_event("startup")
async def on_startup():
    if not SLOWAPI_AVAILABLE:
        LOGGER.warning(
            "slowapi is not installed; using the built-in in-memory rate limiter fallback."
        )
    await warm_result_cache_from_disk()


@app.on_event("shutdown")
async def on_shutdown():
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return TEMPLATES.TemplateResponse(
        "prototype.html",
        build_prototype_context(request),
    )


@app.get("/prototype", response_class=HTMLResponse)
async def prototype(request: Request):
    return TEMPLATES.TemplateResponse(
        "prototype.html",
        build_prototype_context(request),
    )


@app.get("/classic", response_class=HTMLResponse)
async def classic_home(request: Request):
    return TEMPLATES.TemplateResponse(
        "home.html",
        build_home_context(request),
    )


@app.get("/analyze", response_class=HTMLResponse)
@limiter.limit(RATE_LIMIT)
async def analyze_session(
    request: Request,
    input: str,
    session_name: str = "New Session",
    mode: str = "game",
    analysis_profile: str = "preview",
    player_id: str = "",
    account_id: str = "",
):
    form = {
        "session_name": session_name,
        "mode": mode,
        "analysis_profile": analysis_profile,
        "player_id": player_id,
        "account_id": account_id,
        "input": input,
    }

    try:
        normalized_input, cached_result, job = await submit_or_get_job(
            input,
            session_name,
            mode,
            player_id,
            account_id,
            analysis_profile,
        )
    except Exception as exc:
        safe_error = build_safe_request_error(exc)
        return TEMPLATES.TemplateResponse(
            "prototype.html",
            build_prototype_context(
                request,
                bootstrap={
                    "error": safe_error,
                    "form": form,
                },
            ),
            status_code=400,
        )

    if cached_result is not None:
        return render_dashboard(
            request,
            cached_result,
            session_name,
            mode,
            player_id,
            account_id,
            normalized_input,
            analysis_profile,
        )

    return TEMPLATES.TemplateResponse(
        "processing.html",
        {
            "request": request,
            "job_id": job.job_id,
            "session_name": session_name,
            "mode": mode,
            "analysis_profile": analysis_profile,
            "analysis_profile_label": ANALYSIS_PROFILES[analysis_profile]["label"],
            "player_id": player_id.strip(),
            "account_id": account_id.strip(),
            "account_name": get_workout_account_label(account_id),
            "input_source": normalized_input,
            "progress": job.progress,
            "progress_message": job.progress_message,
            "processed_frames": job.processed_frames,
            "total_frames": job.total_frames,
            "processed_chunks": job.processed_chunks,
            "total_chunks": job.total_chunks,
            "partial_outputs": job.partial_outputs,
        },
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def analysis_result(request: Request, job_id: str):
    job = await get_job(job_id)
    if job is None:
        return TEMPLATES.TemplateResponse(
            "home.html",
            build_home_context(request, error="Analysis job not found."),
            status_code=404,
        )

    if job.status == "completed" and job.result is not None:
        return render_dashboard(
            request,
            job.result,
            job.session_name,
            job.mode,
            job.player_id,
            job.account_id,
            job.input_source,
            job.analysis_profile,
        )

    if job.status == "failed":
        return TEMPLATES.TemplateResponse(
            "home.html",
            build_home_context(
                request,
                error=job.error or "Analysis failed.",
                form={
                    "session_name": job.session_name,
                    "mode": job.mode,
                    "analysis_profile": job.analysis_profile,
                    "player_id": job.player_id,
                    "account_id": job.account_id,
                    "input": job.input_source,
                },
            ),
            status_code=500,
        )

    return TEMPLATES.TemplateResponse(
        "processing.html",
        {
            "request": request,
            "job_id": job.job_id,
            "session_name": job.session_name,
            "mode": job.mode,
            "analysis_profile": job.analysis_profile,
            "analysis_profile_label": ANALYSIS_PROFILES[job.analysis_profile]["label"],
            "player_id": job.player_id.strip(),
            "account_id": job.account_id.strip(),
            "account_name": get_workout_account_label(job.account_id),
            "input_source": job.input_source,
            "progress": job.progress,
            "progress_message": job.progress_message,
            "processed_frames": job.processed_frames,
            "total_frames": job.total_frames,
            "processed_chunks": job.processed_chunks,
            "total_chunks": job.total_chunks,
            "partial_outputs": job.partial_outputs,
        },
    )


@app.get("/api/analyze")
@limiter.limit(RATE_LIMIT)
async def analyze_session_api(
    request: Request,
    input: str,
    session_name: str = "New Session",
    mode: str = "game",
    analysis_profile: str = "preview",
    player_id: str = "",
    account_id: str = "",
):
    try:
        normalized_input, cached_result, job = await submit_or_get_job(
            input,
            session_name,
            mode,
            player_id,
            account_id,
            analysis_profile,
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "failed",
                "error": build_safe_request_error(exc),
            },
            status_code=400,
        )
    if cached_result is not None:
        payload = cached_result.to_public_dict()
        payload["session_name"] = session_name
        payload["mode"] = mode
        payload["analysis_profile"] = analysis_profile
        payload["player_id"] = player_id
        payload["account_id"] = account_id
        payload["account_name"] = get_workout_account_label(account_id)
        payload["input"] = normalized_input
        payload["output_url"] = f"/outputs/{cached_result.output_path.name}"
        payload["status"] = "completed"
        payload["cached"] = True
        return JSONResponse(payload)

    return JSONResponse(
        {
            "status": job.status,
            "job_id": job.job_id,
            "poll_url": f"/api/jobs/{job.job_id}",
            "result_url": f"/results/{job.job_id}",
            "analysis_profile": analysis_profile,
            "account_id": account_id,
            "account_name": get_workout_account_label(account_id),
            "input": normalized_input,
        },
        status_code=202,
    )


@app.get("/api/jobs/{job_id}")
@limiter.limit(RATE_LIMIT)
async def get_job_status(request: Request, job_id: str):
    job = await get_job(job_id)
    if job is None:
        return JSONResponse({"status": "missing"}, status_code=404)

    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "session_name": job.session_name,
        "mode": job.mode,
        "analysis_profile": job.analysis_profile,
        "player_id": job.player_id,
        "account_id": job.account_id,
        "account_name": get_workout_account_label(job.account_id),
        "input": job.input_source,
        "result_url": f"/results/{job.job_id}",
        "progress": job.progress,
        "progress_message": job.progress_message,
        "processed_frames": job.processed_frames,
        "total_frames": job.total_frames,
        "processed_chunks": job.processed_chunks,
        "total_chunks": job.total_chunks,
        "partial_outputs": job.partial_outputs,
    }

    if job.status == "failed":
        payload["error"] = job.error
        return JSONResponse(payload, status_code=500)

    if job.status == "completed" and job.result is not None:
        payload.update(job.result.to_public_dict())
        payload["output_url"] = f"/outputs/{job.result.output_path.name}"

    return JSONResponse(payload)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/favicon.ico")
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def quiet_browser_icon_requests():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("live_main:app", host="127.0.0.1", port=8000, reload=False)
