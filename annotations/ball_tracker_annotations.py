from .utils import draw_triangle


class BallTrackerAnnotations:
    def __init__(self):
        self.ball_pointer_color = (0, 250, 0)

    def annotations(self, video_frames, tracker):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracker[frame_num]

            for _, track in ball_dict.items():
                bbox = track.get("bbox") or track.get("box")
                if bbox is None:
                    continue

                frame = draw_triangle(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(frame)

        return output_video_frames
