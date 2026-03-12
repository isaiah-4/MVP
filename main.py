from utils import read_vid, save_vid
from trackers import PlayerTracker, ballTracker
from annotations import (
    PlayerTrackerAnnotations
    )


def main(): 
    #read video
    video_frames = read_vid("Input_vids/video_1.mp4")

    #track players
    player_tracker = PlayerTracker("Models/Player_detection_model.pt")
    ball_tracker = ballTracker("Models/ball_detector_model.pt")


    #run trackers
    player_tracker = player_tracker.get_object_tracks(video_frames, 
                                                      read_from_stub=True, 
                                                      stub_path = "stubs/player_tracker_stub.pkl"
                                                      )
    
    ball_tracker = ball_tracker.get_object_tracks(video_frames,
                                                  read_from_stub=True,
                                                  stub_path = "stubs/ball_tracker_stub.pkl"
                                                  )


    print(player_tracker)

    #draw annotations
    player_tracker_annotations = PlayerTrackerAnnotations(player_tracker)

    output_video_frames = player_tracker_annotations.annotations(video_frames, player_tracker)


    #save video
    save_vid(output_video_frames, "Output_vids/video_1_output.mp4")
if __name__ == "__main__":
    main()
    
