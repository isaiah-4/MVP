from collections import defaultdict

import numpy as np


class SpeedDistanceCalculator:
    def __init__(self, fps=24.0, max_step_per_frame_m=1.0, speed_window=5):
        self.fps = float(fps)
        self.max_step_per_frame_m = float(max_step_per_frame_m)
        self.speed_window = int(speed_window)

    def calculate(self, player_positions_m):
        previous_positions = {}
        previous_frames = {}
        total_distances = {}
        speed_histories = defaultdict(list)
        player_distances_per_frame = []
        player_speeds_per_frame = []

        for frame_num, frame_positions in enumerate(player_positions_m):
            frame_distances = {}
            frame_speeds = {}

            for player_id, position in frame_positions.items():
                position_array = np.asarray(position, dtype=float)
                total_distances.setdefault(player_id, 0.0)

                if player_id in previous_positions:
                    frame_gap = max(1, frame_num - previous_frames[player_id])
                    previous_position = previous_positions[player_id]
                    step_distance = float(
                        np.linalg.norm(position_array - previous_position)
                    )
                    max_allowed_step = self.max_step_per_frame_m * frame_gap

                    if step_distance <= max_allowed_step:
                        total_distances[player_id] += step_distance
                        delta_time_seconds = frame_gap / self.fps
                        speed_kmh = (step_distance / delta_time_seconds) * 3.6
                        speed_histories[player_id].append(speed_kmh)
                        speed_histories[player_id] = speed_histories[player_id][
                            -self.speed_window:
                        ]
                        frame_speeds[player_id] = float(
                            sum(speed_histories[player_id])
                            / len(speed_histories[player_id])
                        )

                previous_positions[player_id] = position_array
                previous_frames[player_id] = frame_num
                frame_distances[player_id] = float(total_distances[player_id])

            player_distances_per_frame.append(frame_distances)
            player_speeds_per_frame.append(frame_speeds)

        return {
            "player_distances_per_frame": player_distances_per_frame,
            "player_speeds_per_frame": player_speeds_per_frame,
            "total_distances": total_distances,
        }
