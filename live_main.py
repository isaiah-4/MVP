from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import shutil
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analysis_pipeline import (
    AnalysisRunResult,
    RESULT_CACHE_VERSION,
    run_analysis,
    run_chunked_full_analysis,
)
from utils.input_utils import extract_youtube_id, is_youtube_url


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))

app = FastAPI(title="CourtVision MVP", version="0.1.0")
app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="assets")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "Output_vids")), name="outputs")

DEFAULT_MAX_WORKERS = max(1, int(os.environ.get("COURTVISION_MAX_WORKERS", "1")))
JOB_TTL_SECONDS = int(os.environ.get("COURTVISION_JOB_TTL_SECONDS", str(6 * 60 * 60)))
RESULT_CACHE_TTL_SECONDS = int(
    os.environ.get("COURTVISION_RESULT_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)
MAX_COMPLETED_JOBS = int(os.environ.get("COURTVISION_MAX_COMPLETED_JOBS", "100"))
MAX_CACHED_RESULTS = int(os.environ.get("COURTVISION_MAX_CACHED_RESULTS", "64"))

EXECUTOR = ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS)
JOB_LOCK = Lock()
RESULT_CACHE = {}
RESULT_CACHE_UPDATED_AT = {}
ACTIVE_JOBS_BY_KEY = {}
JOBS = {}
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


@dataclass
class AnalysisJob:
    job_id: str
    cache_key: str
    input_source: str
    session_name: str
    mode: str
    player_id: str
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


def build_home_context(request, **overrides):
    context = {
        "request": request,
        "error": None,
        "form": {
            "session_name": "Lakers Demo Session",
            "mode": "game",
            "analysis_profile": "preview",
            "player_id": "",
            "input": "Input_vids/video_1.mp4",
        },
        "available_now": [
            "Processed feed with player, ball, court, and tactical overlays",
            "Team-scoped player labels with appearance-based identity cleanup",
            "Touch counts from confirmed ball possession changes",
            "Average possession time per tracked player",
            "Team pass and interception counts",
            "First-pass shot attempts and makes/misses from ball, hoop, and release heuristics",
        ],
        "coming_next": [
            "Shot chart and hot zone views from detected attempt locations",
            "Clip review and side-by-side comparisons",
            "Stronger trained ReID and jersey-number models instead of the current fallback identity layer",
            "AI review and form feedback",
        ],
    }
    context.update(overrides)
    return context


def get_analysis_profile_config(analysis_profile):
    if analysis_profile not in ANALYSIS_PROFILES:
        raise ValueError(f"Unknown analysis profile: {analysis_profile}")
    return ANALYSIS_PROFILES[analysis_profile]


def normalize_input_key(input_source, analysis_profile, mode, player_id):
    source = input_source.strip()
    profile_key = (
        f"profile:{analysis_profile}:mode:{mode}:player:{player_id.strip()}:analysis:{RESULT_CACHE_VERSION}"
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


def validate_input_source(input_source):
    source = input_source.strip()
    if is_youtube_url(source):
        if shutil.which("yt-dlp") is None and importlib.util.find_spec("yt_dlp") is None:
            raise RuntimeError(
                "yt-dlp is required to analyze YouTube URLs. Install it before submitting the job."
            )
        return

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")


def render_dashboard(
    request,
    result,
    session_name,
    mode,
    player_id,
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


def submit_or_get_job(input_source, session_name, mode, player_id, analysis_profile):
    get_analysis_profile_config(analysis_profile)
    validate_input_source(input_source)
    cache_key = normalize_input_key(input_source, analysis_profile, mode, player_id)

    with JOB_LOCK:
        prune_state_locked()
        cached_result = RESULT_CACHE.get(cache_key)
        if cached_result is not None:
            return cached_result, None

        active_job_id = ACTIVE_JOBS_BY_KEY.get(cache_key)
        if active_job_id is not None:
            return None, JOBS[active_job_id]

        job_id = uuid4().hex
        job = AnalysisJob(
            job_id=job_id,
            cache_key=cache_key,
            input_source=input_source,
            session_name=session_name,
            mode=mode,
            player_id=player_id,
            analysis_profile=analysis_profile,
            created_at=time(),
            updated_at=time(),
        )
        JOBS[job_id] = job
        ACTIVE_JOBS_BY_KEY[cache_key] = job_id

    EXECUTOR.submit(run_job, job_id)
    return None, job


def run_job(job_id):
    with JOB_LOCK:
        job = JOBS[job_id]
        job.status = "running"
        job.progress_message = "Starting analysis"
        job.updated_at = time()

    try:
        profile_config = get_analysis_profile_config(job.analysis_profile)
        if job.analysis_profile == "full":
            result = run_chunked_full_analysis(
                job.input_source,
                run_suffix=profile_config["run_suffix"],
                chunk_frames=profile_config.get("chunk_frames", 300),
                max_dimension=profile_config.get("max_dimension"),
                court_keypoint_interval=profile_config.get("court_keypoint_interval", 12),
                mode=job.mode,
                workout_player_id=job.player_id,
                progress_callback=lambda payload: update_job_progress(job_id, payload),
            )
        else:
            update_job_progress(
                job_id,
                {
                    "progress": 0.0,
                    "progress_message": "Analyzing preview",
                },
            )
            result = run_analysis(
                job.input_source,
                max_dimension=profile_config.get("max_dimension"),
                max_frames=profile_config["max_frames"],
                run_suffix=profile_config["run_suffix"],
                court_keypoint_interval=profile_config.get("court_keypoint_interval", 12),
                mode=job.mode,
                workout_player_id=job.player_id,
                progress_callback=lambda payload: update_job_progress(job_id, payload),
            )
    except Exception as exc:
        with JOB_LOCK:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = time()
            ACTIVE_JOBS_BY_KEY.pop(job.cache_key, None)
        return

    with JOB_LOCK:
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


def update_job_progress(job_id, payload):
    with JOB_LOCK:
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


def get_job(job_id):
    with JOB_LOCK:
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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return TEMPLATES.TemplateResponse(
        "home.html",
        build_home_context(request),
    )


@app.get("/analyze", response_class=HTMLResponse)
def analyze_session(
    request: Request,
    input: str,
    session_name: str = "New Session",
    mode: str = "game",
    analysis_profile: str = "preview",
    player_id: str = "",
):
    form = {
        "session_name": session_name,
        "mode": mode,
        "analysis_profile": analysis_profile,
        "player_id": player_id,
        "input": input,
    }

    try:
        cached_result, job = submit_or_get_job(
            input,
            session_name,
            mode,
            player_id,
            analysis_profile,
        )
    except Exception as exc:
        return TEMPLATES.TemplateResponse(
            "home.html",
            build_home_context(request, error=str(exc), form=form),
            status_code=400,
        )

    if cached_result is not None:
        return render_dashboard(
            request,
            cached_result,
            session_name,
            mode,
            player_id,
            input,
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
            "input_source": input,
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
def analysis_result(request: Request, job_id: str):
    job = get_job(job_id)
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
def analyze_session_api(
    input: str,
    session_name: str = "New Session",
    mode: str = "game",
    analysis_profile: str = "preview",
    player_id: str = "",
):
    try:
        cached_result, job = submit_or_get_job(
            input,
            session_name,
            mode,
            player_id,
            analysis_profile,
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "failed",
                "error": str(exc),
            },
            status_code=400,
        )
    if cached_result is not None:
        payload = cached_result.to_public_dict()
        payload["session_name"] = session_name
        payload["mode"] = mode
        payload["analysis_profile"] = analysis_profile
        payload["player_id"] = player_id
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
        },
        status_code=202,
    )


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return JSONResponse({"status": "missing"}, status_code=404)

    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "session_name": job.session_name,
        "mode": job.mode,
        "analysis_profile": job.analysis_profile,
        "player_id": job.player_id,
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
def health():
    return {"ok": True}


@app.get("/favicon.ico")
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
def quiet_browser_icon_requests():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("live_main:app", host="127.0.0.1", port=8000, reload=False)
