from .utils import draw_ellipse

class PlayerTrackerAnnotations:


    def __init__(self, player_tracker):
        pass

    def annotations(self, video_frames,tracker):


        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()


            player_dict = tracker[frame_num]


            for tracker_id, player in player_dict.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue
                frame = draw_ellipse(frame, bbox, (0, 0, 255), tracker_id=tracker_id)

            output_video_frames.append(frame)


        return output_video_frames
