from collections import Counter

from utils import sort_player_identifier


class SessionMetricsBuilder:
    def __init__(self, fps):
        self.fps = float(fps)

    def build(
        self,
        player_tracks,
        team_assignments,
        possession_data,
        pass_interception_data,
        shot_data=None,
        identity_data=None,
    ):
        shot_data = shot_data or {"events": []}
        identity_data = identity_data or {"players_by_id": {}, "players": []}
        identity_by_player = identity_data.get("players_by_id", {})
        player_metrics = {}
        team_possession_frames = Counter()
        team_shot_totals = Counter()
        team_make_totals = Counter()

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
        total_shot_attempts = 0
        total_made_shots = 0

        for event in shot_data.get("events", []):
            shooter_id = event.get("shooter_id", -1)
            team_id = int(event.get("team_id", -1))
            is_make = event.get("result") == "made"

            if shooter_id not in (-1, None, ""):
                player_entry = self._get_player_entry(
                    player_metrics,
                    shooter_id,
                    team_id,
                )
                player_entry["shot_attempts"] += 1
                if is_make:
                    player_entry["made_shots"] += 1

            if team_id != -1:
                team_shot_totals[team_id] += 1
                if is_make:
                    team_make_totals[team_id] += 1

            total_shot_attempts += 1
            if is_make:
                total_made_shots += 1

        for player_id, metrics in player_metrics.items():
            possession_seconds = metrics["possession_frames"] / self.fps
            average_possession_seconds = 0.0
            if metrics["possessions"] > 0:
                average_possession_seconds = possession_seconds / metrics["possessions"]
            shot_attempts = int(metrics["shot_attempts"])
            made_shots = int(metrics["made_shots"])
            field_goal_percentage = 0.0
            if shot_attempts > 0:
                field_goal_percentage = (made_shots / shot_attempts) * 100.0

            row = {
                "player_id": player_id,
                "display_name": identity_by_player.get(player_id, {}).get("display_name", player_id),
                "display_id": identity_by_player.get(player_id, {}).get("display_id", player_id),
                "jersey_number": identity_by_player.get(player_id, {}).get("jersey_number"),
                "identity_source": identity_by_player.get(player_id, {}).get("identity_source", "team_slot"),
                "team_id": int(metrics["team_id"]),
                "tracked_frames": int(metrics["tracked_frames"]),
                "touches": int(metrics["touches"]),
                "possessions": int(metrics["possessions"]),
                "possession_seconds": float(possession_seconds),
                "average_possession_seconds": float(average_possession_seconds),
                "shot_attempts": shot_attempts,
                "made_shots": made_shots,
                "missed_shots": int(max(shot_attempts - made_shots, 0)),
                "field_goal_percentage": float(field_goal_percentage),
            }
            player_rows.append(row)
            total_touches += row["touches"]
            total_possession_frames += metrics["possession_frames"]

        player_rows.sort(
            key=lambda row: (
                -row["touches"],
                -row["tracked_frames"],
                sort_player_identifier(row["player_id"]),
                str(row["player_id"]),
            )
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
            | {
                int(team_id)
                for team_id in team_shot_totals
                if int(team_id) != -1
            }
        )
        if not team_ids:
            team_ids = [1, 2]

        team_rows = []
        for team_id in team_ids:
            attempts = int(team_shot_totals[team_id])
            made = int(team_make_totals[team_id])
            field_goal_percentage = 0.0
            if attempts > 0:
                field_goal_percentage = (made / attempts) * 100.0
            team_rows.append(
                {
                    "team_id": int(team_id),
                    "passes": int(latest_passes.get(team_id, 0)),
                    "interceptions": int(latest_interceptions.get(team_id, 0)),
                    "possession_seconds": float(team_possession_frames[team_id] / self.fps),
                    "shot_attempts": attempts,
                    "made_shots": made,
                    "missed_shots": int(max(attempts - made, 0)),
                    "field_goal_percentage": float(field_goal_percentage),
                }
            )

        average_touch_length_seconds = 0.0
        if total_touches > 0:
            average_touch_length_seconds = (total_possession_frames / self.fps) / total_touches
        session_field_goal_percentage = 0.0
        if total_shot_attempts > 0:
            session_field_goal_percentage = (total_made_shots / total_shot_attempts) * 100.0

        return {
            "overview": {
                "tracked_players": len(player_rows),
                "total_touches": int(total_touches),
                "average_touch_length_seconds": float(average_touch_length_seconds),
                "source_frames": len(player_tracks),
                "total_shot_attempts": int(total_shot_attempts),
                "total_made_shots": int(total_made_shots),
                "total_missed_shots": int(max(total_shot_attempts - total_made_shots, 0)),
                "field_goal_percentage": float(session_field_goal_percentage),
            },
            "players": player_rows,
            "teams": team_rows,
            "shots": [
                {
                    "frame_num": int(event.get("frame_num", 0)),
                    "release_frame": int(event.get("release_frame", 0)),
                    "result": event.get("result", "missed"),
                    "shooter_id": event.get("shooter_id", -1),
                    "shooter_display_name": identity_by_player.get(
                        event.get("shooter_id", -1),
                        {},
                    ).get("display_name", event.get("shooter_id", -1)),
                    "team_id": int(event.get("team_id", -1)),
                    "shot_position_m": event.get("shot_position_m"),
                }
                for event in shot_data.get("events", [])
            ],
            "identity": {
                "appearance_backend": identity_data.get("appearance_backend", "none"),
                "ocr_backend": identity_data.get("ocr_backend", "none"),
                "resolved_players": int(identity_data.get("resolved_players", len(player_rows))),
                "players_with_numbers": int(identity_data.get("players_with_numbers", 0)),
                "primary_identity": identity_data.get("primary_identity"),
                "players": list(identity_data.get("players", [])),
            },
        }

    def _get_player_entry(self, player_metrics, player_id, team_id):
        if player_id not in player_metrics:
            player_metrics[player_id] = {
                "team_id": int(team_id),
                "tracked_frames": 0,
                "touches": 0,
                "possessions": 0,
                "possession_frames": 0,
                "shot_attempts": 0,
                "made_shots": 0,
            }
        return player_metrics[player_id]
