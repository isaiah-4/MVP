import cv2

from .utils import put_text_with_outline


class CourtKeypointAnnotations:
    def __init__(self):
        self.keypoint_color = (0, 255, 255)

    def annotations(self, video_frames, court_keypoints, copy_frames=True):
        output_video_frames = [frame.copy() for frame in video_frames] if copy_frames else video_frames

        for frame_num, frame in enumerate(video_frames):
            frame = output_video_frames[frame_num]
            frame_keypoints = court_keypoints[frame_num]

            for keypoint_id, point in frame_keypoints.items():
                point_x, point_y = [int(value) for value in point]
                cv2.circle(frame, (point_x, point_y), 5, self.keypoint_color, -1)
                cv2.circle(frame, (point_x, point_y), 7, (0, 0, 0), 1)
                frame = put_text_with_outline(
                    frame,
                    str(keypoint_id),
                    (point_x + 6, point_y - 6),
                    font_scale=0.45,
                    color=(255, 255, 255),
                    thickness=1,
                )

            output_video_frames[frame_num] = frame

        return output_video_frames
