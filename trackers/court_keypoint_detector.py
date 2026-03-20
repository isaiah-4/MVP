import numpy as np
from ultralytics import YOLO

from utils import read_stub, save_stub


class CourtKeypointDetector:
    def __init__(self, model_path, keypoint_confidence=0.25):
        self.model = YOLO(model_path)
        self.keypoint_confidence = float(keypoint_confidence)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for frame_index in range(0, len(frames), batch_size):
            batch_frames = frames[frame_index:frame_index + batch_size]
            batch_detections = self.model.predict(
                batch_frames,
                conf=0.5,
                verbose=False,
            )
            detections.extend(batch_detections)

        return detections

    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        court_keypoints = read_stub(read_from_stub, stub_path)
        if court_keypoints is not None and len(court_keypoints) == len(frames):
            return court_keypoints

        detections = self.detect_frames(frames)
        court_keypoints = []

        for detection in detections:
            if detection.keypoints is None or len(detection.keypoints.xy) == 0:
                court_keypoints.append({})
                continue

            frame_points = detection.keypoints.xy[0].cpu().numpy()
            if detection.keypoints.conf is None:
                frame_confidences = np.ones(len(frame_points), dtype=float)
            else:
                frame_confidences = detection.keypoints.conf[0].cpu().numpy()

            frame_keypoints = {}
            for keypoint_id, (point, confidence) in enumerate(
                zip(frame_points, frame_confidences)
            ):
                point_x, point_y = [float(value) for value in point]
                confidence = float(confidence)

                if not np.isfinite(point_x) or not np.isfinite(point_y):
                    continue
                if not np.isfinite(confidence) or confidence < self.keypoint_confidence:
                    continue
                if point_x <= 0 or point_y <= 0:
                    continue

                frame_keypoints[keypoint_id] = (point_x, point_y)

            court_keypoints.append(frame_keypoints)

        save_stub(stub_path, court_keypoints)
        return court_keypoints
