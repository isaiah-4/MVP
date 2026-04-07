import os

import numpy as np
import supervision as sv
from utils import save_stub, read_stub
from .model_store import get_yolo_model

class PlayerTracker:
    def __init__(self, model_path):
        self.model = get_yolo_model(model_path)
        self.tracker = sv.ByteTrack()
        self.batch_size = max(
            1,
            int(os.environ.get("COURTVISION_PLAYER_BATCH_SIZE", "12")),
        )
        self.confidence = float(
            os.environ.get("COURTVISION_PLAYER_CONFIDENCE", "0.5")
        )

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_detections = self.model.predict(
                batch_frames,
                conf=self.confidence,
                verbose=False,
            )
            detections += batch_detections
        return detections
        
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracker = read_stub(read_from_stub, stub_path)
        if tracker is not None:
            if len(tracker) == len(frames):
                return tracker

        self.tracker.reset()
        detections = self.detect_frames(frames)
        tracker = []
        if not detections:
            save_stub(stub_path, tracker)
            return tracker

        cls_names = detections[0].names
        player_class_ids = {
            int(class_id)
            for class_id, class_name in cls_names.items()
            if any(token in class_name.lower() for token in ("human", "player"))
        }
        player_class_id_array = np.asarray(sorted(player_class_ids), dtype=int)

        for frame_num, detection in enumerate(detections):
            detections_supervision = sv.Detections.from_ultralytics(detection)
            if detections_supervision.class_id is not None:
                if player_class_id_array.size > 0:
                    player_mask = np.isin(
                        detections_supervision.class_id.astype(int),
                        player_class_id_array,
                    )
                else:
                    player_mask = np.zeros(len(detections_supervision), dtype=bool)
                detections_supervision = detections_supervision[player_mask]

            detections_with_tracker = self.tracker.update_with_detections(detections_supervision)

            tracker.append({})

            for frame_detection in detections_with_tracker:
                bbox = frame_detection[0].tolist()
                cls_id = int(frame_detection[3])
                track_id = int(frame_detection[4])

                if cls_id in player_class_ids:
                    tracker[frame_num][track_id] = {"bbox": bbox}


        save_stub(stub_path, tracker)

        return tracker
