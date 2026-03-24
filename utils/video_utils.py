import os
import shutil

import cv2

VIDEO_CODEC_CANDIDATES = ("avc1", "H264", "mp4v")


def read_vid(video_path, max_frames=None, start_frame=0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    start_frame = max(0, int(start_frame or 0))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while True:
        if max_frames is not None and len(frames) >= int(max_frames):
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_fps(video_path, fallback_fps=24.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback_fps

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 1:
        return fallback_fps

    return float(fps)


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(frame_count, 0)


def concatenate_videos(input_paths, output_video_path, fps=None):
    if not input_paths:
        raise ValueError("No input videos were provided for concatenation.")

    if len(input_paths) == 1:
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copyfile(str(input_paths[0]), output_video_path)
        return output_video_path

    first_path = str(input_paths[0])
    first_cap = cv2.VideoCapture(first_path)
    if not first_cap.isOpened():
        raise ValueError(f"Could not open video file: {first_path}")

    frame_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    resolved_fps = float(fps) if fps is not None else float(
        first_cap.get(cv2.CAP_PROP_FPS) or 24.0
    )
    first_cap.release()

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Could not determine video dimensions for concatenation.")

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = _open_video_writer(
        output_video_path,
        resolved_fps,
        (frame_width, frame_height),
    )

    try:
        for input_path in input_paths:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")

            try:
                current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if current_width != frame_width or current_height != frame_height:
                    raise ValueError(
                        "Chunk videos must share the same dimensions for concatenation."
                    )

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        writer.release()

    return output_video_path



def save_vid(output_video_frames, output_video_path, fps=24.0):
    if not output_video_frames:
        raise ValueError("No frames were provided for video output.")

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out = _open_video_writer(
        output_video_path,
        float(fps),
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def _open_video_writer(output_video_path, fps, frame_size):
    for codec_name in VIDEO_CODEC_CANDIDATES:
        if os.path.exists(output_video_path):
            os.remove(output_video_path)

        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer
        writer.release()

    raise ValueError(
        f"Could not open a video writer for {output_video_path} using codecs {VIDEO_CODEC_CANDIDATES}."
    )
