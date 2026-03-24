from collections import Counter


class SessionMetricsBuilder:
    def __init__(self, fps):
        self.fps = float(fps)

    def build(
        self,
        player_tracks,
        team_assignments,
        possession_data,
        pass_interception_data,
    ):
        player_metrics = {}
        team_possession_frames = Counter()

        previous_holder = -1
        possession_start_frame = None

        for frame_num, frame_players in enumerate(player_tracks):
            frame_assignment = team_assignments[frame_num]
            current_holder = possession_data["player"][frame_num]
            current_team = possession_data["team"][frame_num]

            if current_team != -1:
                team_possession_frames[current_team] += 1

            for player_id in frame_players:
                player_entry = self._get_player_entry(
                    player_metrics,
                    player_id,
                    frame_assignment.get(player_id, -1),
                )
                player_entry["tracked_frames"] += 1
                player_entry["team_id"] = frame_assignment.get(
                    player_id,
                    player_entry["team_id"],
                )

            if current_holder != previous_holder:
                if previous_holder != -1 and possession_start_frame is not None:
                    player_entry = self._get_player_entry(
                        player_metrics,
                        previous_holder,
                        -1,
                    )
                    player_entry["possessions"] += 1
                    player_entry["possession_frames"] += frame_num - possession_start_frame

                if current_holder != -1:
                    player_entry = self._get_player_entry(
                        player_metrics,
                        current_holder,
                        frame_assignment.get(current_holder, -1),
                    )
                    player_entry["touches"] += 1
                    possession_start_frame = frame_num
                else:
                    possession_start_frame = None

                previous_holder = current_holder

        if previous_holder != -1 and possession_start_frame is not None:
            player_entry = self._get_player_entry(player_metrics, previous_holder, -1)
            player_entry["possessions"] += 1
            player_entry["possession_frames"] += (
                len(possession_data["player"]) - possession_start_frame
            )

        player_rows = []
        total_touches = 0
        total_possession_frames = 0

        for player_id, metrics in player_metrics.items():
            possession_seconds = metrics["possession_frames"] / self.fps
            average_possession_seconds = 0.0
            if metrics["possessions"] > 0:
                average_possession_seconds = possession_seconds / metrics["possessions"]

            row = {
                "player_id": int(player_id),
                "team_id": int(metrics["team_id"]),
                "tracked_frames": int(metrics["tracked_frames"]),
                "touches": int(metrics["touches"]),
                "possessions": int(metrics["possessions"]),
                "possession_seconds": float(possession_seconds),
                "average_possession_seconds": float(average_possession_seconds),
            }
            player_rows.append(row)
            total_touches += row["touches"]
            total_possession_frames += metrics["possession_frames"]

        player_rows.sort(
            key=lambda row: (-row["touches"], -row["tracked_frames"], row["player_id"])
        )

        latest_passes = {}
        latest_interceptions = {}
        if pass_interception_data["passes_per_frame"]:
            latest_passes = pass_interception_data["passes_per_frame"][-1]
        if pass_interception_data["interceptions_per_frame"]:
            latest_interceptions = pass_interception_data["interceptions_per_frame"][-1]

        team_ids = sorted(
            {
                team_id
                for team_id in list(latest_passes.keys()) + list(latest_interceptions.keys())
                if team_id != -1
            }
            | {
                row["team_id"]
                for row in player_rows
                if row["team_id"] != -1
            }
        )
        if not team_ids:
            team_ids = [1, 2]

        team_rows = []
        for team_id in team_ids:
            team_rows.append(
                {
                    "team_id": int(team_id),
                    "passes": int(latest_passes.get(team_id, 0)),
                    "interceptions": int(latest_interceptions.get(team_id, 0)),
                    "possession_seconds": float(team_possession_frames[team_id] / self.fps),
                }
            )

        average_touch_length_seconds = 0.0
        if total_touches > 0:
            average_touch_length_seconds = (total_possession_frames / self.fps) / total_touches

        return {
            "overview": {
                "tracked_players": len(player_rows),
                "total_touches": int(total_touches),
                "average_touch_length_seconds": float(average_touch_length_seconds),
                "source_frames": len(player_tracks),
            },
            "players": player_rows,
            "teams": team_rows,
        }

    def _get_player_entry(self, player_metrics, player_id, team_id):
        if player_id not in player_metrics:
            player_metrics[player_id] = {
                "team_id": int(team_id),
                "tracked_frames": 0,
                "touches": 0,
                "possessions": 0,
                "possession_frames": 0,
            }
        return player_metrics[player_id]
