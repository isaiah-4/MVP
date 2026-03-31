from .utils import put_text_with_outline
from utils import get_foot_position


class SpeedDistanceAnnotations:
    def annotations(
        self,
        video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speeds_per_frame,
        copy_frames=True,
    ):
        output_video_frames = [frame.copy() for frame in video_frames] if copy_frames else video_frames

        for frame_num, frame in enumerate(video_frames):
            frame = output_video_frames[frame_num]
            frame_tracks = player_tracks[frame_num]
            frame_distances = player_distances_per_frame[frame_num]
            frame_speeds = player_speeds_per_frame[frame_num]

            for player_id, player in frame_tracks.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue

                speed = frame_speeds.get(player_id)
                distance = frame_distances.get(player_id)
                if speed is None and distance is None:
                    continue

                foot_x, foot_y = get_foot_position(bbox)
                text_x = max(8, int(foot_x) - 45)
                text_y = int(foot_y) + 34

                if speed is not None:
                    frame = put_text_with_outline(
                        frame,
                        f"{speed:.2f} km/h",
                        (text_x, text_y),
                        font_scale=0.43,
                        color=(255, 255, 255),
                        thickness=1,
                    )

                if distance is not None:
                    frame = put_text_with_outline(
                        frame,
                        f"{distance:.2f} m",
                        (text_x, text_y + 18),
                        font_scale=0.43,
                        color=(255, 255, 255),
                        thickness=1,
                    )

            output_video_frames[frame_num] = frame

        return output_video_frames
