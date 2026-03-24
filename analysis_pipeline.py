from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from analytics import (
    BallPossessionAnalyzer,
    CourtProjector,
    PassInterceptionDetector,
    SessionMetricsBuilder,
    ShotDetector,
    SpeedDistanceCalculator,
    TeamAssigner,
)
from annotations import (
    BallTrackerAnnotations,
    CourtKeypointAnnotations,
    PassInterceptionAnnotations,
    PlayerTrackerAnnotations,
    SpeedDistanceAnnotations,
    TacticalViewAnnotations,
)
from trackers import CourtKeypointDetector, PlayerTracker, ballTracker
from utils import (
    concatenate_videos,
    get_video_fps,
    get_video_frame_count,
    prepare_video_source,
    read_vid,
    save_vid,
)


@dataclass(frozen=True)
class AnalysisRunResult:
    source_key: str
    base_source_key: str
    input_path: Path
    output_path: Path
    fps: float
    player_model_path: str
    ball_model_path: str
    using_court_model: bool
    court_model_path: str
    court_keypoint_interval: int
    chunk_frames: int | None
    max_frames: int | None
    start_frame: int
    processed_frames: int
    chunk_index: int | None
    chunk_count: int | None
    player_ids_are_chunk_local: bool
    chunk_outputs: tuple[dict, ...]
    session_metrics: dict

    def to_public_dict(self):
        return {
            "source_key": self.source_key,
            "base_source_key": self.base_source_key,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "fps": self.fps,
            "player_model_path": self.player_model_path,
            "ball_model_path": self.ball_model_path,
            "using_court_model": self.using_court_model,
            "court_model_path": self.court_model_path,
            "court_keypoint_interval": self.court_keypoint_interval,
            "chunk_frames": self.chunk_frames,
            "max_frames": self.max_frames,
            "start_frame": self.start_frame,
            "processed_frames": self.processed_frames,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "player_ids_are_chunk_local": self.player_ids_are_chunk_local,
            "chunk_outputs": list(self.chunk_outputs),
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
            player_model_path=payload.get("player_model_path", "Models/player_detector.pt"),
            ball_model_path=payload.get("ball_model_path", "Models/ball_detector_model.pt"),
            using_court_model=bool(payload["using_court_model"]),
            court_model_path=payload["court_model_path"],
            court_keypoint_interval=int(payload.get("court_keypoint_interval", 1)),
            chunk_frames=payload.get("chunk_frames"),
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
            session_metrics=payload["session_metrics"],
        )


def run_analysis(
    input_source,
    *,
    player_model="Models/player_detector.pt",
    ball_model="Models/ball_detector_model.pt",
    court_model="Models/court_keypoint_detector.pt",
    court_keypoint_interval=12,
    use_stubs=True,
    output_path=None,
    max_frames=None,
    start_frame=0,
    run_suffix=None,
):
    video_run = prepare_video_source(input_source, run_suffix=run_suffix)
    resolved_output_path = Path(output_path) if output_path is not None else video_run.output_path
    cache_path = resolved_output_path.with_suffix(".json")
    resolved_start_frame = max(0, int(start_frame or 0))

    if use_stubs:
        cached_result = load_cached_result(
            cache_path=cache_path,
            input_path=video_run.input_path,
            output_path=resolved_output_path,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            chunk_frames=None,
            max_frames=max_frames,
            start_frame=resolved_start_frame,
        )
        if cached_result is not None:
            return cached_result

    video_frames = read_vid(
        str(video_run.input_path),
        max_frames=max_frames,
        start_frame=resolved_start_frame,
    )
    if not video_frames:
        raise ValueError(
            f"No video frames were available from frame {resolved_start_frame}: {video_run.input_path}"
        )
    input_fps = get_video_fps(str(video_run.input_path))

    player_tracker_model = PlayerTracker(player_model)
    ball_tracker_model = ballTracker(ball_model)

    player_tracks = player_tracker_model.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(video_run.player_stub_path),
    )

    ball_tracks, hoop_tracks = ball_tracker_model.get_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=str(video_run.ball_stub_path),
    )
    ball_tracks = ball_tracker_model.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker_model.interpolate_ball_positions(ball_tracks)
    hoop_tracks = ball_tracker_model.interpolate_track_positions(hoop_tracks)

    team_assigner = TeamAssigner()
    team_assignments = team_assigner.assign_teams(video_frames, player_tracks)

    possession_analyzer = BallPossessionAnalyzer()
    possession_data = possession_analyzer.detect_possession(
        player_tracks,
        ball_tracks,
        team_assignments,
    )

    pass_interception_detector = PassInterceptionDetector()
    pass_interception_data = pass_interception_detector.detect(possession_data)

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

    projection_data = court_projector.project_tracks(
        court_keypoints,
        player_tracks,
        ball_tracks,
    )

    shot_detector = ShotDetector()
    shot_data = shot_detector.detect(
        player_tracks,
        ball_tracks,
        hoop_tracks,
        possession_data,
        projection_data["player_positions_m"],
    )

    speed_distance_calculator = SpeedDistanceCalculator(fps=input_fps)
    speed_distance_data = speed_distance_calculator.calculate(
        projection_data["player_positions_m"]
    )

    player_tracker_annotations = PlayerTrackerAnnotations()
    ball_tracker_annotations = BallTrackerAnnotations()
    court_keypoint_annotations = CourtKeypointAnnotations()
    pass_interception_annotations = PassInterceptionAnnotations(
        team_assigner.team_colors
    )
    speed_distance_annotations = SpeedDistanceAnnotations()
    tactical_view_annotations = TacticalViewAnnotations(
        court_projector,
        team_assigner.team_colors,
    )

    output_video_frames = player_tracker_annotations.annotations(video_frames, player_tracks)
    output_video_frames = ball_tracker_annotations.annotations(output_video_frames, ball_tracks)
    output_video_frames = court_keypoint_annotations.annotations(
        output_video_frames,
        court_keypoints,
    )
    output_video_frames = pass_interception_annotations.annotations(
        output_video_frames,
        pass_interception_data,
    )
    output_video_frames = speed_distance_annotations.annotations(
        output_video_frames,
        player_tracks,
        speed_distance_data["player_distances_per_frame"],
        speed_distance_data["player_speeds_per_frame"],
    )
    output_video_frames = tactical_view_annotations.annotations(
        output_video_frames,
        projection_data["player_positions_m"],
        projection_data["ball_positions_m"],
        team_assignments,
        possession_data,
    )

    save_vid(output_video_frames, str(resolved_output_path), fps=input_fps)

    session_metrics_builder = SessionMetricsBuilder(input_fps)
    session_metrics = session_metrics_builder.build(
        player_tracks,
        team_assignments,
        possession_data,
        pass_interception_data,
        shot_data,
    )

    result = AnalysisRunResult(
        source_key=video_run.source_key,
        base_source_key=video_run.base_source_key,
        input_path=video_run.input_path,
        output_path=resolved_output_path,
        fps=input_fps,
        player_model_path=player_model,
        ball_model_path=ball_model,
        using_court_model=using_court_model,
        court_model_path=court_model,
        court_keypoint_interval=int(court_keypoint_interval),
        chunk_frames=None,
        max_frames=None if max_frames is None else int(max_frames),
        start_frame=resolved_start_frame,
        processed_frames=len(video_frames),
        chunk_index=None,
        chunk_count=None,
        player_ids_are_chunk_local=False,
        chunk_outputs=(),
        session_metrics=session_metrics,
    )
    save_cached_result(cache_path, result)
    return result


def run_chunked_full_analysis(
    input_source,
    *,
    player_model="Models/player_detector.pt",
    ball_model="Models/ball_detector_model.pt",
    court_model="Models/court_keypoint_detector.pt",
    court_keypoint_interval=12,
    use_stubs=True,
    output_path=None,
    run_suffix="full",
    chunk_frames=300,
    progress_callback=None,
):
    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be greater than zero.")

    video_run = prepare_video_source(input_source, run_suffix=run_suffix)
    resolved_output_path = Path(output_path) if output_path is not None else video_run.output_path
    cache_path = resolved_output_path.with_suffix(".json")

    if use_stubs:
        cached_result = load_cached_result(
            cache_path=cache_path,
            input_path=video_run.input_path,
            output_path=resolved_output_path,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            chunk_frames=chunk_frames,
            max_frames=None,
            start_frame=0,
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
        chunk_result = run_analysis(
            input_source,
            player_model=player_model,
            ball_model=ball_model,
            court_model=court_model,
            court_keypoint_interval=court_keypoint_interval,
            use_stubs=use_stubs,
            max_frames=chunk_max_frames,
            start_frame=start_frame,
            run_suffix=chunk_suffix,
        )
        chunk_results.append(chunk_result)
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
        using_court_model=any(
            chunk_result.using_court_model for chunk_result in chunk_results
        ),
        court_model_path=court_model,
        court_keypoint_interval=int(court_keypoint_interval),
        chunk_frames=int(chunk_frames),
        max_frames=None,
        start_frame=0,
        processed_frames=processed_frames,
        chunk_index=None,
        chunk_count=total_chunks,
        player_ids_are_chunk_local=total_chunks > 1,
        chunk_outputs=tuple(chunk_outputs),
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
    chunk_frames,
    max_frames,
    start_frame,
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

    if payload.get("court_model_path") != court_model:
        return None
    if payload.get("player_model_path") != player_model:
        return None
    if payload.get("ball_model_path") != ball_model:
        return None
    if int(payload.get("court_keypoint_interval", 1)) != int(court_keypoint_interval):
        return None
    if payload.get("chunk_frames") != (None if chunk_frames is None else int(chunk_frames)):
        return None
    if payload.get("max_frames") != (None if max_frames is None else int(max_frames)):
        return None
    if int(payload.get("start_frame", 0)) != int(start_frame):
        return None
    if "shots" not in payload.get("session_metrics", {}):
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

    for chunk_index, chunk_result in enumerate(chunk_results, start=1):
        metrics = chunk_result.session_metrics
        overview = metrics.get("overview", {})
        source_frames += int(overview.get("source_frames", chunk_result.processed_frames))
        total_shot_attempts += int(overview.get("total_shot_attempts", 0))
        total_made_shots += int(overview.get("total_made_shots", 0))

        for row in metrics.get("players", []):
            prefixed_row = dict(row)
            prefixed_row["player_id"] = f"C{chunk_index}-P{row['player_id']}"
            prefixed_row["chunk_index"] = chunk_index
            player_rows.append(prefixed_row)
            total_touches += int(row.get("touches", 0))
            total_possession_seconds += float(row.get("possession_seconds", 0.0))

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
            if shooter_id != -1:
                prefixed_event["shooter_id"] = f"C{chunk_index}-P{shooter_id}"
            prefixed_event["chunk_index"] = chunk_index
            prefixed_event["frame_num"] = int(chunk_result.start_frame) + int(
                event.get("frame_num", 0)
            )
            prefixed_event["release_frame"] = int(chunk_result.start_frame) + int(
                event.get("release_frame", 0)
            )
            shot_events.append(prefixed_event)

    player_rows.sort(
        key=lambda row: (-int(row["touches"]), -int(row["tracked_frames"]), str(row["player_id"]))
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

    return {
        "overview": {
            "tracked_players": len(player_rows),
            "total_touches": int(total_touches),
            "average_touch_length_seconds": float(average_touch_length_seconds),
            "source_frames": int(source_frames),
            "chunk_count": len(chunk_results),
            "player_ids_are_chunk_local": len(chunk_results) > 1,
            "total_shot_attempts": int(total_shot_attempts),
            "total_made_shots": int(total_made_shots),
            "total_missed_shots": int(max(total_shot_attempts - total_made_shots, 0)),
            "field_goal_percentage": float(session_field_goal_percentage),
        },
        "players": player_rows,
        "teams": team_rows,
        "shots": shot_events,
    }
