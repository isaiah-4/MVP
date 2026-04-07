from utils import (
    calculate_overlap_ratio,
    get_bbox_height,
    get_center_of_bbox,
    point_to_bbox_distance,
)


class BallPossessionAnalyzer:
    def __init__(
        self,
        fps=24.0,
        min_frames_to_confirm=5,
        containment_threshold=0.6,
        max_ball_distance=70.0,
        release_frames=None,
        release_hold_seconds=0.45,
    ):
        self.fps = float(fps)
        self.min_frames_to_confirm = min_frames_to_confirm
        self.containment_threshold = containment_threshold
        self.max_ball_distance = max_ball_distance
        if release_frames is None:
            release_frames = max(
                1,
                int(round(float(release_hold_seconds) * max(self.fps, 1.0))),
            )
        self.release_frames = int(release_frames)

    def detect_possession(
        self,
        player_tracks,
        ball_tracks,
        team_assignments,
        *,
        initial_state=None,
    ):
        raw_possession = []

        for frame_num, frame_players in enumerate(player_tracks):
            ball_bbox = self._get_ball_bbox(ball_tracks[frame_num])
            holder_id = self._get_best_candidate(frame_players, ball_bbox)
            raw_possession.append(holder_id)

        confirmed_possession, smooth_state = self._smooth_possession(
            raw_possession,
            initial_state=initial_state,
        )
        team_possession = []

        for frame_num, frame_players in enumerate(player_tracks):
            holder_id = confirmed_possession[frame_num]
            team_id = -1
            if holder_id != -1:
                team_id = team_assignments[frame_num].get(holder_id, -1)

            team_possession.append(team_id)

            for player_id, player in frame_players.items():
                player["has_ball"] = player_id == holder_id

        return {
            "raw_player": raw_possession,
            "player": confirmed_possession,
            "team": team_possession,
            "state": smooth_state,
        }

    def _get_best_candidate(self, frame_players, ball_bbox):
        if ball_bbox is None:
            return -1

        ball_center = get_center_of_bbox(ball_bbox)
        best_containment_id = -1
        best_containment = 0.0
        best_containment_distance = float("inf")

        best_distance_id = -1
        best_distance = float("inf")

        for player_id, player in frame_players.items():
            player_bbox = player.get("bbox") or player.get("box")
            if player_bbox is None:
                continue

            containment = calculate_overlap_ratio(ball_bbox, player_bbox)
            distance = point_to_bbox_distance(ball_center, player_bbox)

            if containment >= self.containment_threshold:
                if (
                    containment > best_containment
                    or (
                        containment == best_containment
                        and distance < best_containment_distance
                    )
                ):
                    best_containment = containment
                    best_containment_distance = distance
                    best_containment_id = player_id
                continue

            allowed_distance = max(
                self.max_ball_distance,
                float(get_bbox_height(player_bbox)) * 0.35,
            )
            if distance <= allowed_distance and distance < best_distance:
                best_distance = distance
                best_distance_id = player_id

        if best_containment_id != -1:
            return best_containment_id

        return best_distance_id

    def _smooth_possession(self, raw_possession, *, initial_state=None):
        state = initial_state or {}
        confirmed_possession = []
        confirmed_holder = state.get("confirmed_holder", -1)
        candidate_holder = state.get("candidate_holder", -1)
        candidate_frames = int(state.get("candidate_frames", 0))
        missing_frames = int(state.get("missing_frames", 0))

        for holder_id in raw_possession:
            if holder_id == confirmed_holder and holder_id != -1:
                candidate_holder = -1
                candidate_frames = 0
                missing_frames = 0
                confirmed_possession.append(confirmed_holder)
                continue

            if holder_id == -1:
                candidate_holder = -1
                candidate_frames = 0
                missing_frames += 1

                if missing_frames >= self.release_frames:
                    confirmed_holder = -1

                confirmed_possession.append(confirmed_holder)
                continue

            missing_frames = 0

            if holder_id != candidate_holder:
                candidate_holder = holder_id
                candidate_frames = 1
            else:
                candidate_frames += 1

            if candidate_frames >= self.min_frames_to_confirm:
                confirmed_holder = candidate_holder

            confirmed_possession.append(confirmed_holder)

        return confirmed_possession, {
            "confirmed_holder": confirmed_holder,
            "candidate_holder": candidate_holder,
            "candidate_frames": int(candidate_frames),
            "missing_frames": int(missing_frames),
        }

    def _get_ball_bbox(self, ball_track):
        if not ball_track:
            return None

        for track in ball_track.values():
            bbox = track.get("bbox") or track.get("box")
            if bbox is not None:
                return bbox

        return None
