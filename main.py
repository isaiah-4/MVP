import argparse
from pathlib import Path

from analytics import (
    BallPossessionAnalyzer,
    CourtProjector,
    PassInterceptionDetector,
    SpeedDistanceCalculator,
    TeamAssigner,
)
from utils import prepare_video_source, read_vid, save_vid
from utils import get_video_fps
from trackers import CourtKeypointDetector, PlayerTracker, ballTracker
from annotations import (
    BallTrackerAnnotations,
    CourtKeypointAnnotations,
    PassInterceptionAnnotations,
    PlayerTrackerAnnotations,
    SpeedDistanceAnnotations,
    TacticalViewAnnotations,
)


DEFAULT_VIDEO_SOURCE = "Input_vids/video_1.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="Run basketball video tracking.")
    parser.add_argument(
        "--input",
        default=DEFAULT_VIDEO_SOURCE,
        help="Local video path or YouTube URL.",
    )
    parser.add_argument(
        "--player-model",
        default="Models/player_detector.pt",
        help="Path to the player detection model.",
    )
    parser.add_argument(
        "--ball-model",
        default="Models/ball_detector_model.pt",
        help="Path to the ball detection model.",
    )
    parser.add_argument(
        "--court-model",
        default="Models/court_keypoint_detector.pt",
        help="Path to the court keypoint model.",
    )
    parser.add_argument(
        "--no-stubs",
        action="store_true",
        help="Disable cached tracker stubs for this run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_run = prepare_video_source(args.input)

    #read video
    video_frames = read_vid(str(video_run.input_path))
    input_fps = get_video_fps(str(video_run.input_path))

    #track players
    player_tracker_model = PlayerTracker(args.player_model)
    ball_tracker_model = ballTracker(args.ball_model)

    #run trackers
    player_tracks = player_tracker_model.get_object_tracks(
        video_frames,
        read_from_stub=not args.no_stubs,
        stub_path=str(video_run.player_stub_path),
    )

    ball_tracks = ball_tracker_model.get_object_tracks(video_frames,
                                                       read_from_stub=not args.no_stubs,
                                                       stub_path = str(video_run.ball_stub_path)
                                                       )
    ball_tracks = ball_tracker_model.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker_model.interpolate_ball_positions(ball_tracks)

    #analytics
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
    if Path(args.court_model).exists():
        court_keypoint_detector = CourtKeypointDetector(args.court_model)
        court_keypoints = court_keypoint_detector.get_court_keypoints(
            video_frames,
            read_from_stub=not args.no_stubs,
            stub_path=str(video_run.court_stub_path),
        )
        court_keypoints = court_projector.validate_keypoints(court_keypoints)
        using_court_model = True
    projection_data = court_projector.project_tracks(
        court_keypoints,
        player_tracks,
        ball_tracks,
    )

    speed_distance_calculator = SpeedDistanceCalculator(fps=input_fps)
    speed_distance_data = speed_distance_calculator.calculate(
        projection_data["player_positions_m"]
    )

    #draw annotations
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
    #save video
    save_vid(output_video_frames, str(video_run.output_path), fps=input_fps)
    print(f"Output saved to {video_run.output_path}")
    if using_court_model:
        print(f"Court keypoints model: {args.court_model}")
    else:
        print("Court keypoints model not found. Using fallback frame-corner projection.")


if __name__ == "__main__":
    main()
