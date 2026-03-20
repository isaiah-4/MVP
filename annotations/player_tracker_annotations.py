import cv2

from .utils import draw_ellipse
from utils import get_center_of_bbox

class PlayerTrackerAnnotations:


    def __init__(self):
        self.default_player_color = (0, 0, 255)
        self.ball_control_color = (0, 0, 255)

    def annotations(self, video_frames,tracker):


        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()


            player_dict = tracker[frame_num]


            for tracker_id, player in player_dict.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue
                player_color = player.get("team_color", self.default_player_color)
                frame = draw_ellipse(frame, bbox, player_color, tracker_id=tracker_id)

                if player.get("has_ball"):
                    x_center, _ = get_center_of_bbox(bbox)
                    marker_y = max(12, int(bbox[1]) - 12)
                    cv2.circle(
                        frame,
                        (int(x_center), marker_y),
                        8,
                        self.ball_control_color,
                        2,
                    )

            output_video_frames.append(frame)


        return output_video_frames
