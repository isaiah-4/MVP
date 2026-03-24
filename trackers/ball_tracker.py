import numpy as np
import supervision as sv
from utils import get_center_of_bbox, save_stub, read_stub
from .model_store import get_yolo_model


class ballTracker:
    def __init__(self, model_path):
        self.model = get_yolo_model(model_path) 


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5, verbose=False)
            detections += batch_detections
        return detections

    def get_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracker = read_stub(read_from_stub, stub_path)
        if isinstance(tracker, dict):
            ball_tracks = tracker.get("ball")
            hoop_tracks = tracker.get("hoop")
            if self._is_valid_track_list(ball_tracks, frames) and self._is_valid_track_list(
                hoop_tracks,
                frames,
            ):
                return ball_tracks, hoop_tracks

        detections = self.detect_frames(frames)
        track_bundle = self._extract_tracks(detections)
        save_stub(stub_path, track_bundle)
        return track_bundle["ball"], track_bundle["hoop"]

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        ball_tracks, _ = self.get_tracks(
            frames,
            read_from_stub=read_from_stub,
            stub_path=stub_path,
        )
        return ball_tracks

    def get_hoop_tracks(self, frames, read_from_stub=False, stub_path=None):
        _, hoop_tracks = self.get_tracks(
            frames,
            read_from_stub=read_from_stub,
            stub_path=stub_path,
        )
        return hoop_tracks

    def remove_wrong_detections(self, ball_positions):
        max_allowed_distance = 25
        last_good_frame_index = -1

        for frame_index, ball_track in enumerate(ball_positions):
            current_bbox = self._get_ball_bbox(ball_track)
            if current_bbox is None:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = frame_index
                continue

            last_good_bbox = self._get_ball_bbox(ball_positions[last_good_frame_index])
            if last_good_bbox is None:
                last_good_frame_index = frame_index
                continue

            frame_gap = frame_index - last_good_frame_index
            allowed_distance = max_allowed_distance * frame_gap
            traveled_distance = self._get_bbox_distance(current_bbox, last_good_bbox)

            if traveled_distance > allowed_distance:
                ball_positions[frame_index] = {}
                continue

            last_good_frame_index = frame_index

        return ball_positions

    def interpolate_track_positions(self, track_positions):
        if not track_positions:
            return track_positions

        bbox_rows = []
        valid_rows = []

        for frame_index, track_dict in enumerate(track_positions):
            bbox = self._get_ball_bbox(track_dict)
            if bbox is None:
                bbox_rows.append([np.nan, np.nan, np.nan, np.nan])
                continue

            bbox_rows.append([float(value) for value in bbox])
            valid_rows.append(frame_index)

        if not valid_rows:
            return track_positions

        bbox_array = np.asarray(bbox_rows, dtype=float)
        frame_indices = np.arange(len(track_positions), dtype=float)

        for coordinate_index in range(4):
            valid_mask = ~np.isnan(bbox_array[:, coordinate_index])
            valid_indices = frame_indices[valid_mask]
            valid_values = bbox_array[valid_mask, coordinate_index]
            bbox_array[:, coordinate_index] = np.interp(frame_indices, valid_indices, valid_values)

        interpolated_positions = []
        for bbox in bbox_array.tolist():
            interpolated_positions.append({
                0: {"bbox": bbox}
            })

        return interpolated_positions

    def _extract_tracks(self, detections):
        ball_tracks = []
        hoop_tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            ball_class_ids = {
                int(class_id)
                for class_id, class_name in cls_names.items()
                if "ball" in class_name.lower()
            }
            hoop_class_ids = {
                int(class_id)
                for class_id, class_name in cls_names.items()
                if any(token in class_name.lower() for token in ("hoop", "rim"))
            }

            detections_supervision = sv.Detections.from_ultralytics(detection)
            chosen_ball_bbox = None
            chosen_hoop_bbox = None
            max_ball_conf = 0.0
            max_hoop_conf = 0.0

            ball_tracks.append({})
            hoop_tracks.append({})

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                confidence = float(frame_detection[2])
                cls_id = int(frame_detection[3])

                if cls_id in ball_class_ids and confidence >= max_ball_conf:
                    chosen_ball_bbox = bbox
                    max_ball_conf = confidence

                if cls_id in hoop_class_ids and confidence >= max_hoop_conf:
                    chosen_hoop_bbox = bbox
                    max_hoop_conf = confidence

            if chosen_ball_bbox is not None:
                ball_tracks[frame_num][0] = {"bbox": chosen_ball_bbox}
            if chosen_hoop_bbox is not None:
                hoop_tracks[frame_num][0] = {"bbox": chosen_hoop_bbox}

        return {
            "ball": ball_tracks,
            "hoop": hoop_tracks,
        }

    def _is_valid_track_list(self, tracker, frames):
        return isinstance(tracker, list) and len(tracker) == len(frames)

    def interpolate_ball_positions(self, ball_positions):
        return self.interpolate_track_positions(ball_positions)

    def _get_ball_bbox(self, ball_track):
        for track in ball_track.values():
            bbox = track.get("bbox") or track.get("box")
            if bbox is not None:
                return bbox
        return None

    def _get_bbox_distance(self, bbox_a, bbox_b):
        center_a = np.asarray(get_center_of_bbox(bbox_a), dtype=float)
        center_b = np.asarray(get_center_of_bbox(bbox_b), dtype=float)
        return float(np.linalg.norm(center_a - center_b))
    
