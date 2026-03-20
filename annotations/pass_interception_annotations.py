import cv2

from .utils import put_text_with_outline


class PassInterceptionAnnotations:
    def __init__(self, team_colors):
        self.team_colors = team_colors

    def annotations(self, video_frames, pass_interception_data):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            passes = pass_interception_data["passes_per_frame"][frame_num]
            interceptions = pass_interception_data["interceptions_per_frame"][frame_num]
            frame = self._draw_scoreboard(frame, passes, interceptions)
            output_video_frames.append(frame)

        return output_video_frames

    def _draw_scoreboard(self, frame, passes, interceptions):
        overlay = frame.copy()
        top_left = (20, 20)
        bottom_right = (275, 120)
        cv2.rectangle(overlay, top_left, bottom_right, (245, 245, 245), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, top_left, bottom_right, (50, 50, 50), 2)

        frame = put_text_with_outline(
            frame,
            "Passes / Interceptions",
            (32, 48),
            font_scale=0.58,
            color=(35, 35, 35),
            outline_color=(255, 255, 255),
            thickness=1,
        )

        for index, team_id in enumerate((1, 2)):
            team_y = 78 + (index * 24)
            color = self.team_colors.get(team_id, (0, 0, 255))
            cv2.circle(frame, (38, team_y - 5), 7, color, -1)
            frame = put_text_with_outline(
                frame,
                f"Team {team_id}: P {passes.get(team_id, 0)}  I {interceptions.get(team_id, 0)}",
                (54, team_y),
                font_scale=0.52,
                color=(20, 20, 20),
                outline_color=(255, 255, 255),
                thickness=1,
            )

        return frame
