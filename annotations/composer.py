import cv2
import numpy as np

from .utils import draw_ellipse, draw_triangle, put_text_with_outline
from utils import get_center_of_bbox, get_foot_position


def render_all_annotations(
    video_frames,
    *,
    player_tracks,
    ball_tracks,
    court_keypoints,
    pass_interception_data,
    player_distances_per_frame,
    player_speeds_per_frame,
    player_positions_m,
    ball_positions_m,
    team_assignments,
    possession_data,
    court_projector,
    team_colors,
    default_player_color=(0, 0, 255),
    ball_pointer_color=(0, 250, 0),
    ball_control_color=(0, 255, 255),
    keypoint_color=(0, 255, 255),
):
    # Build the tactical court once for the full render pass, then copy per frame.
    court_template = _build_court_template(court_projector)
    output_frames = []
    marker_radius = 8

    for frame_num, source_frame in enumerate(video_frames):
        frame = source_frame.copy()
        holder_id = possession_data["player"][frame_num]
        frame_players = player_tracks[frame_num]

        for tracker_id, player in frame_players.items():
            bbox = player.get("bbox") or player.get("box")
            if bbox is None:
                continue

            player_color = player.get("team_color", default_player_color)
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
                    ball_control_color,
                    2,
                )

        for _, track in ball_tracks[frame_num].items():
            bbox = track.get("bbox") or track.get("box")
            if bbox is None:
                continue
            frame = draw_triangle(frame, bbox, ball_pointer_color)

        for keypoint_id, point in court_keypoints[frame_num].items():
            point_x, point_y = int(point[0]), int(point[1])
            cv2.circle(frame, (point_x, point_y), 5, keypoint_color, -1)
            cv2.circle(frame, (point_x, point_y), 7, (0, 0, 0), 1)
            frame = put_text_with_outline(
                frame,
                str(keypoint_id),
                (point_x + 6, point_y - 6),
                font_scale=0.45,
                color=(255, 255, 255),
                thickness=1,
            )

        passes = pass_interception_data["passes_per_frame"][frame_num]
        interceptions = pass_interception_data["interceptions_per_frame"][frame_num]
        frame = _draw_scoreboard(frame, passes, interceptions, team_colors)

        frame_distances = player_distances_per_frame[frame_num]
        frame_speeds = player_speeds_per_frame[frame_num]
        for player_id, player in frame_players.items():
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

        tactical_frame = court_template.copy()
        frame_positions = player_positions_m[frame_num]
        frame_assignment = team_assignments[frame_num]

        for player_id, meter_position in frame_positions.items():
            team_id = frame_assignment.get(player_id, 1)
            color = team_colors.get(team_id, default_player_color)
            point_x, point_y = court_projector.meter_to_pixel(meter_position)
            cv2.circle(tactical_frame, (point_x, point_y), 8, color, -1)
            cv2.circle(tactical_frame, (point_x, point_y), 10, (0, 0, 0), 1)

            if player_id == holder_id:
                cv2.circle(tactical_frame, (point_x, point_y), 14, (0, 0, 255), 2)

        ball_position = ball_positions_m[frame_num]
        if ball_position is not None:
            ball_x, ball_y = court_projector.meter_to_pixel(ball_position)
            cv2.circle(tactical_frame, (ball_x, ball_y), 5, ball_pointer_color, -1)
            cv2.circle(tactical_frame, (ball_x, ball_y), 7, (0, 0, 0), 1)

        output_frames.append(_append_panel(frame, tactical_frame))

    return output_frames


def _build_court_template(court_projector):
    tactical_frame = court_projector.create_tactical_court()
    for _, point in court_projector.get_tactical_keypoints_px().items():
        cv2.circle(tactical_frame, point, 4, (30, 30, 30), -1)
    return tactical_frame


def _draw_scoreboard(frame, passes, interceptions, team_colors):
    top_left = (20, 20)
    bottom_right = (275, 120)
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (245, 245, 245), -1)
    blended_region = cv2.addWeighted(
        overlay,
        0.72,
        frame[y1:y2, x1:x2],
        0.28,
        0,
    )
    frame[y1:y2, x1:x2] = blended_region
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
        color = team_colors.get(team_id, (0, 0, 255))
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


def _append_panel(frame, tactical_frame):
    frame_height = frame.shape[0]
    tactical_height, tactical_width = tactical_frame.shape[:2]
    scaled_width = int(round((frame_height / tactical_height) * tactical_width))
    resized_panel = cv2.resize(tactical_frame, (scaled_width, frame_height))
    separator = np.full((frame_height, 8, 3), (32, 32, 32), dtype=np.uint8)
    return np.concatenate([frame, separator, resized_panel], axis=1)
