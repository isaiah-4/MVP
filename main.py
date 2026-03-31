import argparse

from analysis_pipeline import run_analysis


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
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="Optionally cap the longest frame side before analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_analysis(
        args.input,
        player_model=args.player_model,
        ball_model=args.ball_model,
        court_model=args.court_model,
        max_dimension=args.max_dimension,
        use_stubs=not args.no_stubs,
    )
    print(f"Output saved to {result.output_path}")
    if result.using_court_model:
        print(f"Court keypoints model: {result.court_model_path}")
    else:
        print("Court keypoints model not found. Using fallback frame-corner projection.")
    print(
        "Tracked players:",
        result.session_metrics["overview"]["tracked_players"],
        "| total touches:",
        result.session_metrics["overview"]["total_touches"],
    )


if __name__ == "__main__":
    main()
