from collections import defaultdict

import numpy as np


class SpeedDistanceCalculator:
    def __init__(
        self,
        fps=24.0,
        max_step_per_frame_m=None,
        max_speed_mps=10.0,
        speed_window=5,
    ):
        self.fps = float(fps)
        self.max_step_per_frame_m = (
            None if max_step_per_frame_m is None else float(max_step_per_frame_m)
        )
        self.max_speed_mps = float(max_speed_mps)
        self.speed_window = int(speed_window)

    def calculate(self, player_positions_m, *, initial_state=None, frame_offset=0):
        state = initial_state or {}
        previous_positions = {
            player_id: np.asarray(position, dtype=float)
            for player_id, position in state.get("previous_positions", {}).items()
        }
        previous_frames = {
            player_id: int(frame_num)
            for player_id, frame_num in state.get("previous_frames", {}).items()
        }
        total_distances = {
            player_id: float(distance)
            for player_id, distance in state.get("total_distances", {}).items()
        }
        speed_histories = defaultdict(
            list,
            {
                player_id: [float(value) for value in history]
                for player_id, history in state.get("speed_histories", {}).items()
            },
        )
        player_distances_per_frame = []
        player_speeds_per_frame = []

        for local_frame_num, frame_positions in enumerate(player_positions_m):
            frame_num = int(frame_offset) + int(local_frame_num)
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
                    max_allowed_step = self._get_max_allowed_step(frame_gap)

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
            "state": {
                "previous_positions": {
                    player_id: position.tolist()
                    for player_id, position in previous_positions.items()
                },
                "previous_frames": {
                    player_id: int(frame_num)
                    for player_id, frame_num in previous_frames.items()
                },
                "total_distances": {
                    player_id: float(distance)
                    for player_id, distance in total_distances.items()
                },
                "speed_histories": {
                    player_id: [float(value) for value in history]
                    for player_id, history in speed_histories.items()
                },
            },
        }

    def _get_max_allowed_step(self, frame_gap):
        per_frame_cap = self.max_step_per_frame_m
        if per_frame_cap is None:
            per_frame_cap = self.max_speed_mps / max(self.fps, 1.0)
        return float(per_frame_cap) * max(1, int(frame_gap))
