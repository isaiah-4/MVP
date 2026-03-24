import numpy as np

from utils import read_stub, save_stub
from .model_store import get_yolo_model


class CourtKeypointDetector:
    def __init__(self, model_path, keypoint_confidence=0.25, frame_interval=12):
        self.model = get_yolo_model(model_path)
        self.keypoint_confidence = float(keypoint_confidence)
        self.frame_interval = max(1, int(frame_interval))

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

        if not frames:
            return []

        sampled_indices = self._build_sampled_indices(len(frames))
        sampled_frames = [frames[frame_index] for frame_index in sampled_indices]
        detections = self.detect_frames(sampled_frames)
        sampled_keypoints = [
            self._extract_frame_keypoints(detection)
            for detection in detections
        ]
        court_keypoints = self._expand_sampled_keypoints(
            sampled_indices,
            sampled_keypoints,
            len(frames),
        )

        save_stub(stub_path, court_keypoints)
        return court_keypoints

    def _build_sampled_indices(self, frame_count):
        sampled_indices = list(range(0, frame_count, self.frame_interval))
        if not sampled_indices:
            return [0]
        if sampled_indices[-1] != frame_count - 1:
            sampled_indices.append(frame_count - 1)
        return sampled_indices

    def _extract_frame_keypoints(self, detection):
        if detection.keypoints is None or len(detection.keypoints.xy) == 0:
            return {}

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

        return frame_keypoints

    def _expand_sampled_keypoints(self, sampled_indices, sampled_keypoints, frame_count):
        if frame_count == 0:
            return []
        if len(sampled_indices) == 1:
            return [dict(sampled_keypoints[0]) for _ in range(frame_count)]

        expanded_keypoints = [{} for _ in range(frame_count)]

        for sample_position in range(len(sampled_indices) - 1):
            left_frame_index = sampled_indices[sample_position]
            right_frame_index = sampled_indices[sample_position + 1]
            left_keypoints = sampled_keypoints[sample_position]
            right_keypoints = sampled_keypoints[sample_position + 1]
            frame_span = max(1, right_frame_index - left_frame_index)

            for frame_index in range(left_frame_index, right_frame_index):
                interpolation_alpha = (frame_index - left_frame_index) / frame_span
                expanded_keypoints[frame_index] = self._interpolate_frame_keypoints(
                    left_keypoints,
                    right_keypoints,
                    interpolation_alpha,
                )

        expanded_keypoints[sampled_indices[-1]] = dict(sampled_keypoints[-1])
        return expanded_keypoints

    def _interpolate_frame_keypoints(
        self,
        left_keypoints,
        right_keypoints,
        interpolation_alpha,
    ):
        if not left_keypoints and not right_keypoints:
            return {}
        if not left_keypoints:
            return dict(right_keypoints)
        if not right_keypoints:
            return dict(left_keypoints)

        interpolation_alpha = float(np.clip(interpolation_alpha, 0.0, 1.0))
        frame_keypoints = {}
        shared_ids = set(left_keypoints) & set(right_keypoints)

        for keypoint_id in shared_ids:
            left_point = np.asarray(left_keypoints[keypoint_id], dtype=float)
            right_point = np.asarray(right_keypoints[keypoint_id], dtype=float)
            interpolated_point = left_point + ((right_point - left_point) * interpolation_alpha)
            frame_keypoints[keypoint_id] = (
                float(interpolated_point[0]),
                float(interpolated_point[1]),
            )

        nearest_keypoints = left_keypoints if interpolation_alpha <= 0.5 else right_keypoints
        for keypoint_id, point in nearest_keypoints.items():
            if keypoint_id in frame_keypoints:
                continue
            frame_keypoints[keypoint_id] = (float(point[0]), float(point[1]))

        return frame_keypoints
