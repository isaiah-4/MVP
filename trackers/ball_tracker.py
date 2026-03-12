from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append("../utils")
from utils import read_vid, save_vid, save_stub, read_stub


class ballTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
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

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[1]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv["ball"]:
                    if max_conf < confidence:
                        chosenBBox = bbox
                        max_conf = confidence

            if chosenBBox is not None:
                tracker[frame_num][0] = {"bbox": chosenBBox}
            



        save_stub(stub_path, tracker)
        
        return tracker
    
