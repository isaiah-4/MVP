from ultralytics import YOLO
import numpy as np
import supervision as sv
import sys 
sys.path.append("../utils")
from utils import get_center_of_bbox, save_stub, read_stub


class ballTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5, verbose=False)
            detections += batch_detections
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        tracker = read_stub(read_from_stub, stub_path)
        if tracker is not None:
            if len(tracker) == len(frames):
                return tracker
            
        detections = self.detect_frames(frames)
        tracker = []

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}


            detections_supervision = sv.Detections.from_ultralytics(detection)

            tracker.append({})
            chosenBBox = None
            max_conf = 0
            ball_class_ids = {
                class_id for class_name, class_id in cls_names_inv.items()
                if "ball" in class_name.lower()
            }

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                confidence = float(frame_detection[2])
                cls_id = int(frame_detection[3])

                if cls_id in ball_class_ids:
                    if max_conf < confidence:
                        chosenBBox = bbox
                        max_conf = confidence

            if chosenBBox is not None:
                tracker[frame_num][0] = {"bbox": chosenBBox}
            



        save_stub(stub_path, tracker)
        
        return tracker

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

    def interpolate_ball_positions(self, ball_positions):
        if not ball_positions:
            return ball_positions

        bbox_rows = []
        valid_rows = []

        for frame_index, ball_track in enumerate(ball_positions):
            bbox = self._get_ball_bbox(ball_track)
            if bbox is None:
                bbox_rows.append([np.nan, np.nan, np.nan, np.nan])
                continue

            bbox_rows.append([float(value) for value in bbox])
            valid_rows.append(frame_index)

        if not valid_rows:
            return ball_positions

        bbox_array = np.asarray(bbox_rows, dtype=float)
        frame_indices = np.arange(len(ball_positions), dtype=float)

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
    
