import cv2
import os



def read_vid(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames=[]
    while True:
        ret,frame = cap.read()
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



def save_vid(output_video_frames, output_video_path, fps=24.0):
    if not output_video_frames:
        raise ValueError("No frames were provided for video output.")

    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))      

    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
