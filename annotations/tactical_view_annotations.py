import cv2
import numpy as np


class TacticalViewAnnotations:
    def __init__(self, court_projector, team_colors):
        self.court_projector = court_projector
        self.team_colors = team_colors

    def annotations(
        self,
        video_frames,
        player_positions_m,
        ball_positions_m,
        team_assignments,
        possession_data,
    ):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            tactical_frame = self.court_projector.create_tactical_court()
            frame_positions = player_positions_m[frame_num]
            frame_assignment = team_assignments[frame_num]
            holder_id = possession_data["player"][frame_num]

            for player_id, meter_position in frame_positions.items():
                team_id = frame_assignment.get(player_id, 1)
                color = self.team_colors.get(team_id, (0, 0, 255))
                point_x, point_y = self.court_projector.meter_to_pixel(meter_position)
                cv2.circle(tactical_frame, (point_x, point_y), 8, color, -1)
                cv2.circle(tactical_frame, (point_x, point_y), 10, (0, 0, 0), 1)

                if player_id == holder_id:
                    cv2.circle(tactical_frame, (point_x, point_y), 14, (0, 0, 255), 2)

            ball_position = ball_positions_m[frame_num]
            if ball_position is not None:
                ball_x, ball_y = self.court_projector.meter_to_pixel(ball_position)
                cv2.circle(tactical_frame, (ball_x, ball_y), 5, (0, 250, 0), -1)
                cv2.circle(tactical_frame, (ball_x, ball_y), 7, (0, 0, 0), 1)

            tactical_frame = self._draw_tactical_keypoints(tactical_frame)
            combined_frame = self._append_panel(frame, tactical_frame)
            output_video_frames.append(combined_frame)

        return output_video_frames

    def _draw_tactical_keypoints(self, tactical_frame):
        for _, point in self.court_projector.get_tactical_keypoints_px().items():
            point_x, point_y = point
            cv2.circle(tactical_frame, (point_x, point_y), 4, (30, 30, 30), -1)
        return tactical_frame

    def _append_panel(self, frame, tactical_frame):
        frame_height = frame.shape[0]
        tactical_height, tactical_width = tactical_frame.shape[:2]
        scaled_width = int(round((frame_height / tactical_height) * tactical_width))
        resized_panel = cv2.resize(tactical_frame, (scaled_width, frame_height))
        separator = np.full((frame_height, 8, 3), (32, 32, 32), dtype=np.uint8)
        return np.concatenate([frame, separator, resized_panel], axis=1)
