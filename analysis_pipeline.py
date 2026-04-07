from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from analytics import (
    BallPossessionAnalyzer,
    CourtProjector,
    PassInterceptionDetector,
    PlayerIdentityResolver,
    SessionMetricsBuilder,
    ShotDetector,
    SpeedDistanceCalculator,
    TeamAssigner,
)
from annotations import (
    render_all_annotations,
)
from trackers import CourtKeypointDetector, PlayerTracker, ballTracker
from utils import (
    concatenate_videos,
    get_video_fps,
    get_video_frame_count,
    normalize_player_track_ids_by_team,
    prepare_video_source,
    read_vid,
    save_vid,
    sanitize_name,
    sort_player_identifier,
)


RESULT_CACHE_VERSION = 9


@dataclass(frozen=True)
class AnalysisRunResult:
    source_key: str
    base_source_key: str
    input_path: Path
    output_path: Path
    fps: float
    player_model_path: str
    ball_model_path: str
    model_file_state: dict
    using_court_model: bool
    court_model_path: str
    court_keypoint_interval: int
    mode: str
    workout_player_id: str
    chunk_frames: int | None
    max_dimension: int | None
    max_frames: int | None
    start_frame: int
    processed_frames: int
    chunk_index: int | None
    chunk_count: int | None
    player_ids_are_chunk_local: bool
    chunk_outputs: tuple[dict, ...]
    carry_state: dict
    session_metrics: dict

    def to_public_dict(self):
        return {
            "cache_version": RESULT_CACHE_VERSION,
            "source_key": self.source_key,
            "base_source_key": self.base_source_key,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "fps": self.fps,
            "player_model_path": self.player_model_path,
            "ball_model_path": self.ball_model_path,
            "model_file_state": self.model_file_state,
            "using_court_model": self.using_court_model,
            "court_model_path": self.court_model_path,
            "court_keypoint_interval": self.court_keypoint_interval,
            "mode": self.mode,
            "workout_player_id": self.workout_player_id,
            "chunk_frames": self.chunk_frames,
            "max_dimension": self.max_dimension,
            "max_frames": self.max_frames,
            "start_frame": self.start_frame,
            "processed_frames": self.processed_frames,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "player_ids_are_chunk_local": self.player_ids_are_chunk_local,
            "chunk_outputs": list(self.chunk_outputs),
            "carry_state": self.carry_state,
            "session_metrics": self.session_metrics,
        }

    @classmethod
    def from_public_dict(cls, payload):
        return cls(
            source_key=payload["source_key"],
            base_source_key=payload.get("base_source_key", payload["source_key"]),
            input_path=Path(payload["input_path"]),
            output_path=Path(payload["output_path"]),
            fps=float(payload["fps"]),
            player_model_path=payload.get("player_model_path", "Models/Player_detection_model.pt"),
            ball_model_path=payload.get("ball_model_path", "Models/ball_detector_model.pt"),
            model_file_state=payload.get("model_file_state", {}),
            using_court_model=bool(payload["using_court_model"]),
            court_model_path=payload["court_model_path"],
            court_keypoint_interval=int(payload.get("court_keypoint_interval", 1)),
            mode=str(payload.get("mode", "game")),
            workout_player_id=str(payload.get("workout_player_id", "")),
            chunk_frames=payload.get("chunk_frames"),
            max_dimension=payload.get("max_dimension"),
            max_frames=payload.get("max_frames"),
            start_frame=int(payload.get("start_frame", 0)),
            processed_frames=int(
                payload.get(
                    "processed_frames",
                    payload.get("session_metrics", {})
                    .get("overview", {})
                    .get("source_frames", 0),
                )
            ),
            chunk_index=payload.get("chunk_index"),
            chunk_count=payload.get("chunk_count"),
            player_ids_are_chunk_local=bool(
                payload.get("player_ids_are_chunk_local", False)
            ),
            chunk_outputs=tuple(payload.get("chunk_outputs", [])),
            carry_state=payload.get("carry_state", {}),
            session_metrics=payload["session_metrics"],
        )


def _build_effective_run_suffix(run_suffix, max_dimension, mode, workout_player_id):
    suffix_parts = []
    if run_suffix:
        suffix_parts.append(run_suffix)
    if max_dimension is not None:
        suffix_parts.append(f"{int(max_dimension)}px")
    if mode and str(mode) != "game":
        suffix_parts.append(str(mode))
    if workout_player_id:
        suffix_parts.append(sanitize_name(str(workout_player_id)))
    if not suffix_parts:
        return None
    return "_".join(suffix_parts)


def _build_model_file_state(*, player_model, ball_model, court_model):
    return {
        "player_model": _get_model_file_state(player_model),
        "ball_model": _get_model_file_state(ball_model),
        "court_model": _get_model_file_state(court_model),
    }


def _get_model_file_state(model_path):
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        return {
            "path": str(resolved_path),
            "exists": False,
            "mtime_ns": None,
            "size": None,
        }

    stat_result = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "exists": True,
        "mtime_ns": int(stat_result.st_mtime_ns),
        "size": int(stat_result.st_size),
    }


def _append_metric_warning(session_metrics, *, code, message):
    warnings = session_metrics.setdefault("warnings", [])
    if any(existing.get("code") == code for existing in warnings):
        return
    warnings.append(
        {
            "code": str(code),
            "message": str(message),
        }
    )


def run_analysis(
    input_source,
    *,
    player_model="Models/Player_detection_model.pt",
    ball_model="Models/ball_detector_model.pt",
    court_model="Models/court_keypoint_detector.pt",
    court_keypoint_interval=12,
    mode="game",
    workout_player_id="",
    use_stubs=True,
    output_path=None,
    max_dimension=None,
    max_frames=None,
    start_frame=2,
    run_suffix=None,
    initial_carry_state=None,
    finalize_open_session_state=True,
    progress_callback=None,
):
    effective_run_suffix = _build_effective_run_suffix(
        run_suffix,
        max_dimension,
        mode,
        workout_player_id,
    )
    video_run = prepare_video_source(input_source, run_suffix=effective_run_suffix)
    resolved_output_path = Path(output_path) if output_path is not None else video_run.output_path
    cache_path = resolved_output_path.with_suffix(".json")
    resolved_start_frame = max(0, int(start_frame or 0))
    local_total_frames = None
    initial_carry_state = dict(initial_carry_state or {})
    model_file_state = _build_model_file_state(
        player_model=player_model,
        ball_model=ball_model,
        court_model=court_model,
    )

    def emit_progress(progress, message):
        if progress_callback is None:
            return

        payload = {
            "progress": float(progress),
            "progress_message": str(message),
        }
        if local_total_frames is not None:
            payload["total_frames"] = int(local_total_frames)
            payload["processed_frames"] = min(
                int(round(local_total_frames * float(progress))),
                int(local_total_frames),
            )
        progress_callback(payload)

    if use_stubs:
        cached_result = load_cached_result(
            cache_path=cache_path,
            input_path=video_run.input_path,
            output_path=resolved_output_path,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            mode=mode,
            workout_player_id=workout_player_id,
            chunk_frames=None,
            max_dimension=max_dimension,
            max_frames=max_frames,
            start_frame=resolved_start_frame,
            model_file_state=model_file_state,
        )
        if cached_result is not None:
            local_total_frames = cached_result.processed_frames
            emit_progress(1.0, "Loaded cached result")
            return cached_result

    emit_progress(0.01, "Loading video frames")
    video_frames = read_vid(
        str(video_run.input_path),
        max_dimension=max_dimension,
        max_frames=max_frames,
        start_frame=resolved_start_frame,
    )
    if not video_frames:
        raise ValueError(
            f"No video frames were available from frame {resolved_start_frame}: {video_run.input_path}"
        )
    local_total_frames = len(video_frames)
    emit_progress(0.05, "Loaded video frames")
    input_fps = get_video_fps(str(video_run.input_path))

    player_tracker_model = PlayerTracker(player_model)
    ball_tracker_model = ballTracker(ball_model)

    raw_player_tracks = player_tracker_model.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(video_run.player_stub_path),
    )
    emit_progress(0.2, "Detected players")

    ball_tracks, hoop_tracks = ball_tracker_model.get_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(video_run.ball_stub_path),
    )
    ball_tracks = ball_tracker_model.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker_model.interpolate_ball_positions(ball_tracks)
    hoop_tracks = ball_tracker_model.interpolate_track_positions(hoop_tracks)
    emit_progress(0.38, "Detected ball and hoop")

    team_assigner = TeamAssigner()
    raw_team_assignments = team_assigner.assign_teams(video_frames, raw_player_tracks)
    player_tracks, team_assignments = normalize_player_track_ids_by_team(
        raw_player_tracks,
        raw_team_assignments,
        team_colors=team_assigner.team_colors,
        max_players_per_team=5,
    )
    identity_resolver = PlayerIdentityResolver()
    identity_resolution = identity_resolver.resolve(
        video_frames,
        player_tracks,
        team_assignments,
        mode=mode,
        workout_player_id=workout_player_id,
    )
    player_tracks = identity_resolution["player_tracks"]
    team_assignments = identity_resolution["team_assignments"]
    identity_data = identity_resolution["identity_data"]
    emit_progress(0.48, "Assigned teams and resolved identities")

    possession_analyzer = BallPossessionAnalyzer(fps=input_fps)
    possession_data = possession_analyzer.detect_possession(
        player_tracks,
        ball_tracks,
        team_assignments,
        initial_state=initial_carry_state.get("possession"),
    )

    pass_interception_detector = PassInterceptionDetector()
    pass_interception_data = pass_interception_detector.detect(
        possession_data,
        initial_state=initial_carry_state.get("passes"),
    )
    emit_progress(0.56, "Computed possession and passing")

    court_projector = CourtProjector()
    court_keypoints = court_projector.detect_keypoints(video_frames)
    using_court_model = False

    if Path(court_model).exists():
        court_keypoint_detector = CourtKeypointDetector(
            court_model,
            frame_interval=court_keypoint_interval,
        )
        court_keypoints = court_keypoint_detector.get_court_keypoints(
            video_frames,
            read_from_stub=use_stubs,
            stub_path=str(video_run.court_stub_path),
        )
        court_keypoints = court_projector.validate_keypoints(court_keypoints)
        using_court_model = True
    emit_progress(0.68, "Projected court geometry")

    projection_data = court_projector.project_tracks(
        court_keypoints,
        player_tracks,
        ball_tracks,
    )
    projection_available = any(
        bool(frame_positions)
        for frame_positions in projection_data["player_positions_m"]
    )

    shot_detector = ShotDetector(fps=input_fps)
    shot_data = shot_detector.detect(
        player_tracks,
        ball_tracks,
        hoop_tracks,
        possession_data,
        projection_data["player_positions_m"],
    )

    speed_distance_calculator = SpeedDistanceCalculator(fps=input_fps)
    if projection_available:
        speed_distance_data = speed_distance_calculator.calculate(
            projection_data["player_positions_m"],
            initial_state=initial_carry_state.get("movement"),
            frame_offset=resolved_start_frame,
        )
    else:
        movement_state = dict(initial_carry_state.get("movement") or {})
        movement_state["previous_positions"] = {}
        movement_state["previous_frames"] = {}
        speed_distance_data = {
            "player_distances_per_frame": [{} for _ in video_frames],
            "player_speeds_per_frame": [{} for _ in video_frames],
            "total_distances": {},
            "state": movement_state,
        }
    emit_progress(0.76, "Computed shot and movement analytics")

    output_video_frames = render_all_annotations(
        video_frames,
        player_tracks=player_tracks,
        ball_tracks=ball_tracks,
        court_keypoints=court_keypoints,
        pass_interception_data=pass_interception_data,
        player_distances_per_frame=speed_distance_data["player_distances_per_frame"],
        player_speeds_per_frame=speed_distance_data["player_speeds_per_frame"],
        player_positions_m=projection_data["player_positions_m"],
        ball_positions_m=projection_data["ball_positions_m"],
        team_assignments=team_assignments,
        possession_data=possession_data,
        court_projector=court_projector,
        team_colors=team_assigner.team_colors,
    )
    emit_progress(0.9, "Rendered overlays")

    save_vid(output_video_frames, str(resolved_output_path), fps=input_fps)
    emit_progress(0.97, "Saved processed video")

    session_metrics_builder = SessionMetricsBuilder(input_fps)
    session_metrics = session_metrics_builder.build(
        player_tracks,
        team_assignments,
        possession_data,
        pass_interception_data,
        shot_data,
        identity_data=identity_data,
        initial_state=initial_carry_state.get("session_metrics"),
        frame_offset=resolved_start_frame,
        finalize_open_state=bool(finalize_open_session_state),
    )
    session_metrics_state = session_metrics.pop("state", {})
    session_metrics.setdefault("overview", {})["projection_available"] = bool(
        projection_available
    )
    if not projection_available:
        _append_metric_warning(
            session_metrics,
            code="court_projection_unavailable",
            message=(
                "Court projection was unavailable for this run, so speed, distance, "
                "shot chart, and hot zone data are limited to detections with valid coordinates."
            ),
        )

    result = AnalysisRunResult(
        source_key=video_run.source_key,
        base_source_key=video_run.base_source_key,
        input_path=video_run.input_path,
        output_path=resolved_output_path,
        fps=input_fps,
        player_model_path=player_model,
        ball_model_path=ball_model,
        model_file_state=model_file_state,
        using_court_model=bool(using_court_model and projection_available),
        court_model_path=court_model,
        court_keypoint_interval=int(court_keypoint_interval),
        mode=str(mode),
        workout_player_id=str(workout_player_id or ""),
        chunk_frames=None,
        max_dimension=None if max_dimension is None else int(max_dimension),
        max_frames=None if max_frames is None else int(max_frames),
        start_frame=resolved_start_frame,
        processed_frames=len(video_frames),
        chunk_index=None,
        chunk_count=None,
        player_ids_are_chunk_local=False,
        chunk_outputs=(),
        carry_state={
            "possession": dict(possession_data.get("state", {})),
            "passes": dict(pass_interception_data.get("state", {})),
            "movement": dict(speed_distance_data.get("state", {})),
            "session_metrics": dict(session_metrics_state),
        },
        session_metrics=session_metrics,
    )
    save_cached_result(cache_path, result)
    emit_progress(1.0, "Completed")
    return result


def run_chunked_full_analysis(
    input_source,
    *,
    player_model="Models/Player_detection_model.pt",
    ball_model="Models/ball_detector_model.pt",
    court_model="Models/court_keypoint_detector.pt",
    court_keypoint_interval=12,
    mode="game",
    workout_player_id="",
    use_stubs=True,
    output_path=None,
    run_suffix="full",
    chunk_frames=300,
    max_dimension=None,
    progress_callback=None,
):
    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be greater than zero.")

    effective_run_suffix = _build_effective_run_suffix(
        run_suffix,
        max_dimension,
        mode,
        workout_player_id,
    )
    video_run = prepare_video_source(input_source, run_suffix=effective_run_suffix)
    resolved_output_path = Path(output_path) if output_path is not None else video_run.output_path
    cache_path = resolved_output_path.with_suffix(".json")
    model_file_state = _build_model_file_state(
        player_model=player_model,
        ball_model=ball_model,
        court_model=court_model,
    )

    if use_stubs:
        cached_result = load_cached_result(
            cache_path=cache_path,
            input_path=video_run.input_path,
            output_path=resolved_output_path,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            mode=mode,
            workout_player_id=workout_player_id,
            chunk_frames=chunk_frames,
            max_dimension=max_dimension,
            max_frames=None,
            start_frame=0,
            model_file_state=model_file_state,
        )
        if cached_result is not None:
            return cached_result

    total_frames = get_video_frame_count(str(video_run.input_path))
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames: {video_run.input_path}")

    input_fps = get_video_fps(str(video_run.input_path))
    chunk_frames = int(chunk_frames)
    total_chunks = ((total_frames - 1) // chunk_frames) + 1
    chunk_results = []
    chunk_outputs = []
    processed_frames = 0
    carry_state = {}

    if progress_callback is not None:
        progress_callback(
            {
                "progress": 0.0,
                "progress_message": f"Queued {total_chunks} chunk(s)",
                "processed_chunks": 0,
                "total_chunks": total_chunks,
                "processed_frames": 0,
                "total_frames": total_frames,
                "partial_outputs": [],
            }
        )

    for chunk_index, start_frame in enumerate(range(0, total_frames, chunk_frames), start=1):
        chunk_max_frames = min(chunk_frames, total_frames - start_frame)
        chunk_suffix = f"{run_suffix}_chunk_{chunk_index:03d}"
        processed_frames_before_chunk = processed_frames
        is_last_chunk = chunk_index == total_chunks

        def chunk_progress(local_payload):
            if progress_callback is None:
                return

            local_progress = float(local_payload.get("progress", 0.0))
            local_processed_frames = local_payload.get("processed_frames")
            if local_processed_frames is None:
                local_processed_frames = int(round(chunk_max_frames * local_progress))
            overall_processed_frames = min(
                processed_frames_before_chunk + int(local_processed_frames),
                total_frames,
            )
            progress_callback(
                {
                    "progress": min(overall_processed_frames / total_frames, 0.999),
                    "progress_message": (
                        f"Chunk {chunk_index} of {total_chunks}: "
                        f"{local_payload.get('progress_message', 'Processing')}"
                    ),
                    "processed_chunks": chunk_index - 1,
                    "total_chunks": total_chunks,
                    "processed_frames": overall_processed_frames,
                    "total_frames": total_frames,
                    "partial_outputs": list(chunk_outputs),
                }
            )

        chunk_result = run_analysis(
            input_source,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            mode=mode,
            workout_player_id=workout_player_id,
            use_stubs=use_stubs,
            max_dimension=max_dimension,
            max_frames=chunk_max_frames,
            start_frame=start_frame,
            run_suffix=chunk_suffix,
            initial_carry_state=carry_state,
            finalize_open_session_state=is_last_chunk,
            progress_callback=chunk_progress,
        )
        chunk_results.append(chunk_result)
        carry_state = dict(chunk_result.carry_state)
        processed_frames += chunk_result.processed_frames

        chunk_record = {
            "chunk_index": chunk_index,
            "chunk_count": total_chunks,
            "start_frame": start_frame,
            "frame_count": chunk_result.processed_frames,
            "output_path": str(chunk_result.output_path),
            "output_name": chunk_result.output_path.name,
            "output_url": f"/outputs/{chunk_result.output_path.name}",
        }
        chunk_outputs.append(chunk_record)

        if progress_callback is not None:
            progress_callback(
                {
                    "progress": min(processed_frames / total_frames, 1.0),
                    "progress_message": f"Processed chunk {chunk_index} of {total_chunks}",
                    "processed_chunks": chunk_index,
                    "total_chunks": total_chunks,
                    "processed_frames": processed_frames,
                    "total_frames": total_frames,
                    "partial_outputs": list(chunk_outputs),
                }
            )

    if progress_callback is not None:
        progress_callback(
            {
                "progress": 1.0,
                "progress_message": "Stitching final video",
                "processed_chunks": total_chunks,
                "total_chunks": total_chunks,
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "partial_outputs": list(chunk_outputs),
            }
        )

    concatenate_videos(
        [chunk_result.output_path for chunk_result in chunk_results],
        str(resolved_output_path),
        fps=input_fps,
    )

    result = AnalysisRunResult(
        source_key=video_run.source_key,
        base_source_key=video_run.base_source_key,
        input_path=video_run.input_path,
        output_path=resolved_output_path,
        fps=input_fps,
        player_model_path=player_model,
        ball_model_path=ball_model,
        model_file_state=model_file_state,
        using_court_model=any(
            chunk_result.using_court_model for chunk_result in chunk_results
        ),
        court_model_path=court_model,
        court_keypoint_interval=int(court_keypoint_interval),
        mode=str(mode),
        workout_player_id=str(workout_player_id or ""),
        chunk_frames=int(chunk_frames),
        max_dimension=None if max_dimension is None else int(max_dimension),
        max_frames=None,
        start_frame=0,
        processed_frames=processed_frames,
        chunk_index=None,
        chunk_count=total_chunks,
        player_ids_are_chunk_local=total_chunks > 1,
        chunk_outputs=tuple(chunk_outputs),
        carry_state={},
        session_metrics=combine_chunk_session_metrics(chunk_results),
    )
    save_cached_result(cache_path, result)
    return result


def load_cached_result(
    *,
    cache_path,
    input_path,
    output_path,
    player_model,
    ball_model,
    court_model,
    court_keypoint_interval,
    mode,
    workout_player_id,
    chunk_frames,
    max_dimension,
    max_frames,
    start_frame,
    model_file_state,
):
    if not cache_path.exists() or not output_path.exists():
        return None

    try:
        input_mtime_ns = input_path.stat().st_mtime_ns
        output_mtime_ns = output_path.stat().st_mtime_ns
        cache_mtime_ns = cache_path.stat().st_mtime_ns
    except FileNotFoundError:
        return None

    if output_mtime_ns < input_mtime_ns or cache_mtime_ns < input_mtime_ns:
        return None

    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if int(payload.get("cache_version", 0)) != RESULT_CACHE_VERSION:
        return None
    if payload.get("court_model_path") != court_model:
        return None
    if payload.get("player_model_path") != player_model:
        return None
    if payload.get("ball_model_path") != ball_model:
        return None
    if payload.get("model_file_state", {}) != model_file_state:
        return None
    if str(payload.get("mode", "game")) != str(mode):
        return None
    if str(payload.get("workout_player_id", "")) != str(workout_player_id or ""):
        return None
    if int(payload.get("court_keypoint_interval", 1)) != int(court_keypoint_interval):
        return None
    if payload.get("chunk_frames") != (None if chunk_frames is None else int(chunk_frames)):
        return None
    if payload.get("max_dimension") != (None if max_dimension is None else int(max_dimension)):
        return None
    if payload.get("max_frames") != (None if max_frames is None else int(max_frames)):
        return None
    if int(payload.get("start_frame", 0)) != int(start_frame):
        return None

    return AnalysisRunResult.from_public_dict(payload)


def save_cached_result(cache_path, result):
    cache_path.write_text(json.dumps(result.to_public_dict(), indent=2))


def combine_chunk_session_metrics(chunk_results):
    player_rows = []
    team_totals = defaultdict(
        lambda: {
            "passes": 0,
            "interceptions": 0,
            "possession_seconds": 0.0,
            "shot_attempts": 0,
            "made_shots": 0,
        }
    )
    source_frames = 0
    total_touches = 0
    total_possession_seconds = 0.0
    total_shot_attempts = 0
    total_made_shots = 0
    shot_events = []
    identity_players = []
    warnings = []

    for chunk_index, chunk_result in enumerate(chunk_results, start=1):
        metrics = chunk_result.session_metrics
        overview = metrics.get("overview", {})
        for warning in metrics.get("warnings", []):
            if any(existing.get("code") == warning.get("code") for existing in warnings):
                continue
            warnings.append(dict(warning))
        source_frames += int(overview.get("source_frames", chunk_result.processed_frames))
        total_shot_attempts += int(overview.get("total_shot_attempts", 0))
        total_made_shots += int(overview.get("total_made_shots", 0))

        for row in metrics.get("players", []):
            prefixed_row = dict(row)
            prefixed_row["player_id"] = f"C{chunk_index}-{row['player_id']}"
            prefixed_row["display_name"] = (
                f"C{chunk_index}-{row.get('display_name', row['player_id'])}"
            )
            prefixed_row["chunk_index"] = chunk_index
            player_rows.append(prefixed_row)
            total_touches += int(row.get("touches", 0))
            total_possession_seconds += float(row.get("possession_seconds", 0.0))

        for identity_row in metrics.get("identity", {}).get("players", []):
            prefixed_identity = dict(identity_row)
            prefixed_identity["player_id"] = f"C{chunk_index}-{identity_row['player_id']}"
            prefixed_identity["display_name"] = (
                f"C{chunk_index}-{identity_row.get('display_name', identity_row['player_id'])}"
            )
            prefixed_identity["chunk_index"] = chunk_index
            identity_players.append(prefixed_identity)

        for row in metrics.get("teams", []):
            team_id = int(row.get("team_id", -1))
            if team_id == -1:
                continue
            team_totals[team_id]["passes"] += int(row.get("passes", 0))
            team_totals[team_id]["interceptions"] += int(row.get("interceptions", 0))
            team_totals[team_id]["possession_seconds"] += float(
                row.get("possession_seconds", 0.0)
            )
            team_totals[team_id]["shot_attempts"] += int(row.get("shot_attempts", 0))
            team_totals[team_id]["made_shots"] += int(row.get("made_shots", 0))

        for event in metrics.get("shots", []):
            prefixed_event = dict(event)
            shooter_id = event.get("shooter_id", -1)
            if shooter_id not in (-1, None, ""):
                prefixed_event["shooter_id"] = f"C{chunk_index}-{shooter_id}"
            shooter_display_name = event.get("shooter_display_name")
            if shooter_display_name not in (None, ""):
                prefixed_event["shooter_display_name"] = f"C{chunk_index}-{shooter_display_name}"
            prefixed_event["chunk_index"] = chunk_index
            prefixed_event["frame_num"] = int(chunk_result.start_frame) + int(
                event.get("frame_num", 0)
            )
            prefixed_event["release_frame"] = int(chunk_result.start_frame) + int(
                event.get("release_frame", 0)
            )
            shot_events.append(prefixed_event)

    player_rows.sort(
        key=lambda row: (
            -int(row["touches"]),
            -int(row["tracked_frames"]),
            sort_player_identifier(row["player_id"]),
            str(row["player_id"]),
        )
    )

    team_rows = [
        {
            "team_id": int(team_id),
            "passes": int(team_totals[team_id]["passes"]),
            "interceptions": int(team_totals[team_id]["interceptions"]),
            "possession_seconds": float(team_totals[team_id]["possession_seconds"]),
            "shot_attempts": int(team_totals[team_id]["shot_attempts"]),
            "made_shots": int(team_totals[team_id]["made_shots"]),
            "missed_shots": int(
                max(team_totals[team_id]["shot_attempts"] - team_totals[team_id]["made_shots"], 0)
            ),
            "field_goal_percentage": float(
                (
                    (team_totals[team_id]["made_shots"] / team_totals[team_id]["shot_attempts"]) * 100.0
                )
                if team_totals[team_id]["shot_attempts"] > 0
                else 0.0
            ),
        }
        for team_id in sorted(team_totals)
    ]
    
    if not team_rows:
        team_rows = [
            {
                "team_id": 1,
                "passes": 0,
                "interceptions": 0,
                "possession_seconds": 0.0,
                "shot_attempts": 0,
                "made_shots": 0,
                "missed_shots": 0,
                "field_goal_percentage": 0.0,
            },
            {
                "team_id": 2,
                "passes": 0,
                "interceptions": 0,
                "possession_seconds": 0.0,
                "shot_attempts": 0,
                "made_shots": 0,
                "missed_shots": 0,
                "field_goal_percentage": 0.0,
            },
        ]

    average_touch_length_seconds = 0.0
    if total_touches > 0:
        average_touch_length_seconds = total_possession_seconds / total_touches
    session_field_goal_percentage = 0.0
    if total_shot_attempts > 0:
        session_field_goal_percentage = (total_made_shots / total_shot_attempts) * 100.0

    if len(chunk_results) > 1:
        warnings.append(
            {
                "code": "chunk_local_player_ids",
                "message": (
                    "This run was processed in chunks, so player labels are chunk-local "
                    "until cross-chunk identity matching is implemented."
                ),
            }
        )

    return {
        "overview": {
            "tracked_players": len(player_rows),
            "total_touches": int(total_touches),
            "average_touch_length_seconds": float(average_touch_length_seconds),
            "source_frames": int(source_frames),
            "chunk_count": len(chunk_results),
            "player_ids_are_chunk_local": len(chunk_results) > 1,
            "projection_available": all(
                bool(
                    chunk_result.session_metrics.get("overview", {}).get(
                        "projection_available",
                        False,
                    )
                )
                for chunk_result in chunk_results
            ),
            "total_shot_attempts": int(total_shot_attempts),
            "total_made_shots": int(total_made_shots),
            "total_missed_shots": int(max(total_shot_attempts - total_made_shots, 0)),
            "field_goal_percentage": float(session_field_goal_percentage),
        },
        "players": player_rows,
        "teams": team_rows,
        "shots": shot_events,
        "warnings": warnings,
        "identity": {
            "appearance_backend": "chunk_local",
            "ocr_backend": "chunk_local",
            "resolved_players": len(identity_players),
            "players_with_numbers": sum(1 for row in identity_players if row.get("jersey_number")),
            "primary_identity": None,
            "players": identity_players,
        },
    }
