import cv2
import numpy as np

from utils import get_center_of_bbox, get_foot_position


class CourtProjector:
    def __init__(
        self,
        court_width_m=15.0,
        court_length_m=28.0,
        tactical_scale=22,
        panel_padding=18,
        fallback_margin=0.06,
        keypoint_error_tolerance=0.8,
    ):
        self.court_width_m = float(court_width_m)
        self.court_length_m = float(court_length_m)
        self.tactical_scale = int(tactical_scale)
        self.panel_padding = int(panel_padding)
        self.fallback_margin = float(fallback_margin)
        self.keypoint_error_tolerance = float(keypoint_error_tolerance)
        self.target_points_m = {
            0: (0.0, 0.0),
            1: (0.91, 0.0),
            2: (5.18, 0.0),
            3: (10.0, 0.0),
            4: (14.1, 0.0),
            5: (15.0, 0.0),
            6: (15.0, 14.0),
            7: (0.0, 14.0),
            8: (5.18, 5.79),
            9: (10.0, 5.79),
            10: (15.0, 28.0),
            11: (14.1, 28.0),
            12: (10.0, 28.0),
            13: (5.18, 28.0),
            14: (0.91, 28.0),
            15: (0.0, 28.0),
            16: (5.18, 22.21),
            17: (10.0, 22.21),
        }

    def detect_keypoints(self, video_frames):
        court_keypoints = []
        for frame in video_frames:
            frame_height, frame_width = frame.shape[:2]
            margin_x = int(frame_width * self.fallback_margin)
            margin_y = int(frame_height * self.fallback_margin)

            court_keypoints.append(
                {
                    0: (margin_x, margin_y),
                    5: (frame_width - margin_x - 1, margin_y),
                    10: (frame_width - margin_x - 1, frame_height - margin_y - 1),
                    15: (margin_x, frame_height - margin_y - 1),
                }
            )

        return court_keypoints

    def validate_keypoints(self, court_keypoints):
        validated_keypoints = []

        for frame_keypoints in court_keypoints:
            filtered_keypoints = dict(frame_keypoints)
            detected_ids = [
                keypoint_id
                for keypoint_id in filtered_keypoints
                if keypoint_id in self.target_points_m
            ]

            if len(detected_ids) < 4:
                validated_keypoints.append(filtered_keypoints)
                continue

            invalid_ids = set()
            for keypoint_id in detected_ids:
                other_ids = [
                    other_id
                    for other_id in detected_ids
                    if other_id != keypoint_id and other_id not in invalid_ids
                ]
                if len(other_ids) < 2:
                    continue

                compare_id_a, compare_id_b = other_ids[:2]
                detected_distance_a = self._measure_distance(
                    filtered_keypoints[keypoint_id],
                    filtered_keypoints[compare_id_a],
                )
                detected_distance_b = self._measure_distance(
                    filtered_keypoints[keypoint_id],
                    filtered_keypoints[compare_id_b],
                )
                target_distance_a = self._measure_distance(
                    self.target_points_m[keypoint_id],
                    self.target_points_m[compare_id_a],
                )
                target_distance_b = self._measure_distance(
                    self.target_points_m[keypoint_id],
                    self.target_points_m[compare_id_b],
                )

                if detected_distance_b == 0 or target_distance_b == 0:
                    continue

                detected_ratio = detected_distance_a / detected_distance_b
                target_ratio = target_distance_a / target_distance_b
                if target_ratio == 0:
                    continue

                ratio_error = abs((detected_ratio - target_ratio) / target_ratio)
                if ratio_error > self.keypoint_error_tolerance:
                    invalid_ids.add(keypoint_id)

            for invalid_id in invalid_ids:
                filtered_keypoints.pop(invalid_id, None)

            validated_keypoints.append(filtered_keypoints)

        return validated_keypoints

    def project_tracks(self, court_keypoints, player_tracks, ball_tracks):
        player_positions_m = []
        ball_positions_m = []

        for frame_num, frame_players in enumerate(player_tracks):
            homography = self._build_homography(court_keypoints[frame_num])
            frame_player_positions = {}
            frame_ball_position = None

            if homography is not None:
                for player_id, player in frame_players.items():
                    bbox = player.get("bbox") or player.get("box")
                    if bbox is None:
                        continue

                    foot_position = get_foot_position(bbox)
                    projected_position = self._transform_point(homography, foot_position)
                    if projected_position is None:
                        continue

                    frame_player_positions[player_id] = projected_position

                ball_bbox = self._get_ball_bbox(ball_tracks[frame_num])
                if ball_bbox is not None:
                    ball_center = get_center_of_bbox(ball_bbox)
                    frame_ball_position = self._transform_point(homography, ball_center)

            player_positions_m.append(frame_player_positions)
            ball_positions_m.append(frame_ball_position)

        return {
            "player_positions_m": player_positions_m,
            "ball_positions_m": ball_positions_m,
        }

    def create_tactical_court(self):
        width_px = int(self.court_width_m * self.tactical_scale) + (2 * self.panel_padding)
        height_px = int(self.court_length_m * self.tactical_scale) + (2 * self.panel_padding)
        court = np.full((height_px, width_px, 3), (223, 210, 184), dtype=np.uint8)

        top_left = (self.panel_padding, self.panel_padding)
        bottom_right = (
            width_px - self.panel_padding,
            height_px - self.panel_padding,
        )
        cv2.rectangle(court, top_left, bottom_right, (60, 60, 60), 2)

        center_x = int(width_px / 2)
        center_y = int(height_px / 2)
        cv2.line(
            court,
            (self.panel_padding, center_y),
            (width_px - self.panel_padding, center_y),
            (80, 80, 80),
            1,
        )

        center_circle_radius = int(1.8 * self.tactical_scale)
        cv2.circle(
            court,
            (center_x, center_y),
            center_circle_radius,
            (80, 80, 80),
            1,
        )

        lane_left_x = self.meter_to_pixel((5.18, 0.0))[0]
        lane_right_x = self.meter_to_pixel((10.0, 0.0))[0]
        free_throw_top_y = self.meter_to_pixel((0.0, 5.79))[1]
        free_throw_bottom_y = self.meter_to_pixel((0.0, 22.21))[1]
        top_baseline_y = self.panel_padding
        bottom_baseline_y = height_px - self.panel_padding

        cv2.rectangle(
            court,
            (lane_left_x, top_baseline_y),
            (lane_right_x, free_throw_top_y),
            (80, 80, 80),
            1,
        )
        cv2.rectangle(
            court,
            (lane_left_x, free_throw_bottom_y),
            (lane_right_x, bottom_baseline_y),
            (80, 80, 80),
            1,
        )

        cv2.putText(
            court,
            "Tactical View",
            (self.panel_padding, max(22, self.panel_padding - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
        )
        return court

    def meter_to_pixel(self, meter_position):
        x_meter, y_meter = meter_position
        x_pixel = int(round((x_meter * self.tactical_scale) + self.panel_padding))
        y_pixel = int(round((y_meter * self.tactical_scale) + self.panel_padding))
        return x_pixel, y_pixel

    def get_tactical_keypoints_px(self):
        return {
            keypoint_id: self.meter_to_pixel(meter_position)
            for keypoint_id, meter_position in self.target_points_m.items()
        }

    def _build_homography(self, frame_keypoints):
        if not frame_keypoints:
            return None

        valid_ids = [
            keypoint_id
            for keypoint_id in sorted(frame_keypoints)
            if keypoint_id in self.target_points_m
        ]
        if len(valid_ids) < 4:
            return None

        source_points = np.float32([frame_keypoints[keypoint_id] for keypoint_id in valid_ids])
        target_points = np.float32([self.target_points_m[keypoint_id] for keypoint_id in valid_ids])
        homography, _ = cv2.findHomography(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.0,
        )
        return homography

    def _transform_point(self, homography, point):
        source_point = np.asarray([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source_point, homography)
        transformed_x, transformed_y = transformed[0][0]

        if not np.isfinite(transformed_x) or not np.isfinite(transformed_y):
            return None

        return (
            float(np.clip(transformed_x, 0.0, self.court_width_m)),
            float(np.clip(transformed_y, 0.0, self.court_length_m)),
        )

    def _get_ball_bbox(self, ball_track):
        if not ball_track:
            return None

        for track in ball_track.values():
            bbox = track.get("bbox") or track.get("box")
            if bbox is not None:
                return bbox

        return None

    def _measure_distance(self, point_a, point_b):
        point_a = np.asarray(point_a, dtype=float)
        point_b = np.asarray(point_b, dtype=float)
        return float(np.linalg.norm(point_a - point_b))
