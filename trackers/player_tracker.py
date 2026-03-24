import supervision as sv
import sys 
sys.path.append("../utils")
from utils import read_vid, save_vid, save_stub, read_stub
from .model_store import get_yolo_model

class PlayerTracker:
    def __init__(self, model_path):
        self.model = get_yolo_model(model_path)
        self.tracker = sv.ByteTrack()

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
            player_class_ids = {
                class_id
                for class_id, class_name in cls_names.items()
                if any(token in class_name.lower() for token in ("human", "player"))
            }


            detections_supervision = sv.Detections.from_ultralytics(detection)

            detections_with_tracker= self.tracker.update_with_detections(detections_supervision)

            tracker.append({})

            for frame_detection in detections_with_tracker:
                bbox = frame_detection[0].tolist()
                cls_id = int(frame_detection[3])
                track_id = int(frame_detection[4])

                if cls_id in player_class_ids:
                    tracker[frame_num][track_id] = {"bbox": bbox}


        save_stub(stub_path, tracker)

        return tracker
