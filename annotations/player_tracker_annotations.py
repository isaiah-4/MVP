import cv2

from .utils import draw_ellipse
from utils import get_center_of_bbox

class PlayerTrackerAnnotations:


    def __init__(self):
        self.default_player_color = (0, 0, 255)
        self.ball_control_color = (0, 255, 255)

    def annotations(self, video_frames, tracker, copy_frames=True):
        output_video_frames = [frame.copy() for frame in video_frames] if copy_frames else video_frames
        marker_radius = 8
        for frame_num, frame in enumerate(video_frames):
            frame = output_video_frames[frame_num]
            player_dict = tracker[frame_num]
            for tracker_id, player in player_dict.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue
                player_color = player.get("team_color", self.default_player_color)
                frame = draw_ellipse(
                    frame,
                    bbox,
                    player_color,
                    tracker_id=player.get("display_id", tracker_id),
                )

                if player.get("has_ball"):
                    x_center, _ = get_center_of_bbox(bbox)
                    marker_y = max(marker_radius + 2, int(bbox[1]) - 12)
                    cv2.circle(
                        frame,
                        (int(x_center), marker_y),
                        marker_radius,
                        self.ball_control_color,
                        2,
                    )

            output_video_frames[frame_num] = frame
        return output_video_frames
