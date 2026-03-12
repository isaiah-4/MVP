import cv2
import sys
from annotations.utils import draw_triangle
sys.path.append("../trackers")
sys.path.append("../utils")

class ball_Follower_Annotations:
    def __init__(self):
        self.ball_pointer_color = (0, 250, 0)

    def annotations(self, video_frames, ball_tracker):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_frames.append(frame.copy())
            ball_dict = ball_tracker[frame_num]


            for _, track in ball_dict.items():
                bbox = track["bbox"]
                if bbox is not None:
                    continue
                output_frames = draw_triangle(output_frames, frame_num, bbox, self.ball_pointer_color) 
               
            output_frames.append(output_frames)
        return output_frames