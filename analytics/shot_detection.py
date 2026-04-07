from collections import defaultdict

from utils import (
    calculate_overlap_ratio,
    get_bbox_height,
    get_bbox_width,
    get_center_of_bbox,
    point_to_bbox_distance,
)


class ShotDetector:
    def __init__(
        self,
        *,
        fps=24.0,
        max_flight_frames=None,
        max_flight_seconds=1.6,
        cooldown_frames=None,
        cooldown_seconds=0.75,
        min_release_gap_frames=None,
        min_release_gap_seconds=0.08,
        min_candidate_gap_frames=None,
        min_candidate_gap_seconds=0.42,
        rebound_resolution_frames=None,
        rebound_resolution_seconds=0.33,
    ):
        self.fps = float(fps)
        self.max_flight_frames = self._resolve_frame_count(
            max_flight_frames,
            max_flight_seconds,
        )
        self.cooldown_frames = self._resolve_frame_count(
            cooldown_frames,
            cooldown_seconds,
        )
        self.min_release_gap_frames = self._resolve_frame_count(
            min_release_gap_frames,
            min_release_gap_seconds,
        )
        self.min_candidate_gap_frames = self._resolve_frame_count(
            min_candidate_gap_frames,
            min_candidate_gap_seconds,
        )
        self.rebound_resolution_frames = self._resolve_frame_count(
            rebound_resolution_frames,
            rebound_resolution_seconds,
        )

    def detect(
        self,
        player_tracks,
        ball_tracks,
        hoop_tracks,
        possession_data,
        player_positions_m=None,
    ):
        frame_count = min(
            len(player_tracks),
            len(ball_tracks),
            len(hoop_tracks),
            len(possession_data.get("player", [])),
            len(possession_data.get("raw_player", [])),
            len(possession_data.get("team", [])),
        )
        if frame_count == 0:
            empty_counts = []
            return {
                "events": [],
                "events_by_frame": {},
                "attempts_per_frame": empty_counts,
                "makes_per_frame": empty_counts,
                "misses_per_frame": empty_counts,
            }

        if player_positions_m is None:
            player_positions_m = [{} for _ in range(frame_count)]

        events = []
        events_by_frame = defaultdict(list)
        active_candidate = None
        cooldown_until = -1
        missing_raw_frames = 0
        last_candidate_release = -self.min_candidate_gap_frames

        previous_raw_holder = -1
        last_confirmed_holder = -1
        last_confirmed_team = -1
        previous_ball_center = None

        for frame_num in range(frame_count):
            current_holder = possession_data["player"][frame_num]
            current_team = int(possession_data["team"][frame_num])
            raw_holder = possession_data["raw_player"][frame_num]
            ball_bbox = self._get_bbox(ball_tracks[frame_num])
            hoop_bbox = self._get_bbox(hoop_tracks[frame_num])
            ball_center = (
                get_center_of_bbox(ball_bbox)
                if ball_bbox is not None
                else None
            )

            if current_holder != -1:
                last_confirmed_holder = current_holder
                last_confirmed_team = current_team

            if raw_holder == -1:
                missing_raw_frames += 1
            else:
                missing_raw_frames = 0

            if active_candidate is None and frame_num > cooldown_until:
                if (
                    previous_raw_holder != -1
                    and previous_raw_holder == last_confirmed_holder
                    and raw_holder == -1
                    and missing_raw_frames >= self.min_release_gap_frames
                    and (frame_num - last_candidate_release) >= self.min_candidate_gap_frames
                    and self._is_plausible_release(
                        player_tracks[frame_num],
                        ball_bbox,
                        last_confirmed_holder,
                    )
                ):
                    active_candidate = {
                        "release_frame": frame_num,
                        "shooter_id": last_confirmed_holder,
                        "team_id": last_confirmed_team,
                        "approach_frame": None,
                        "release_position_m": self._get_release_position(
                            player_positions_m,
                            frame_num,
                            last_confirmed_holder,
                        ),
                    }
                    last_candidate_release = frame_num

            if active_candidate is not None:
                if ball_bbox is not None and hoop_bbox is not None:
                    if self._ball_is_near_hoop(ball_bbox, hoop_bbox):
                        if active_candidate["approach_frame"] is None:
                            active_candidate["approach_frame"] = frame_num

                if (
                    active_candidate["approach_frame"] is not None
                    and previous_ball_center is not None
                    and ball_center is not None
                    and hoop_bbox is not None
                    and self._ball_crossed_hoop(
                        previous_ball_center,
                        ball_center,
                        hoop_bbox,
                    )
                ):
                    event = self._build_event(
                        frame_num=frame_num,
                        candidate=active_candidate,
                        result="made",
                    )
                    events.append(event)
                    events_by_frame[frame_num].append(event)
                    cooldown_until = frame_num + self.cooldown_frames
                    active_candidate = None
                elif self._candidate_expired(
                    frame_num,
                    current_holder,
                    active_candidate,
                ):
                    if active_candidate["approach_frame"] is not None:
                        event = self._build_event(
                            frame_num=frame_num,
                            candidate=active_candidate,
                            result="missed",
                        )
                        events.append(event)
                        events_by_frame[frame_num].append(event)
                        cooldown_until = frame_num + self.cooldown_frames
                    active_candidate = None

            previous_raw_holder = raw_holder
            previous_ball_center = ball_center

        attempts_per_frame, makes_per_frame, misses_per_frame = self._build_cumulative_counts(
            frame_count,
            events,
        )

        return {
            "events": events,
            "events_by_frame": dict(events_by_frame),
            "attempts_per_frame": attempts_per_frame,
            "makes_per_frame": makes_per_frame,
            "misses_per_frame": misses_per_frame,
        }

    def _candidate_expired(self, frame_num, current_holder, candidate):
        frames_since_release = frame_num - candidate["release_frame"]
        if frames_since_release >= self.max_flight_frames:
            return True

        if (
            current_holder != -1
            and frames_since_release >= self.rebound_resolution_frames
        ):
            return True

        return False

    def _build_event(self, *, frame_num, candidate, result):
        shot_position_m = candidate["release_position_m"]
        if shot_position_m is not None:
            shot_position_m = [
                float(shot_position_m[0]),
                float(shot_position_m[1]),
            ]

        return {
            "frame_num": int(frame_num),
            "release_frame": int(candidate["release_frame"]),
            "result": result,
            "shooter_id": candidate["shooter_id"],
            "team_id": int(candidate["team_id"]),
            "shot_position_m": shot_position_m,
        }

    def _build_cumulative_counts(self, frame_count, events):
        attempts = {1: 0, 2: 0}
        makes = {1: 0, 2: 0}
        misses = {1: 0, 2: 0}

        events_by_frame = defaultdict(list)
        for event in events:
            events_by_frame[int(event["frame_num"])].append(event)

        attempts_per_frame = []
        makes_per_frame = []
        misses_per_frame = []

        for frame_num in range(frame_count):
            for event in events_by_frame.get(frame_num, []):
                team_id = int(event.get("team_id", -1))
                if team_id == -1:
                    continue
                attempts[team_id] = attempts.get(team_id, 0) + 1
                if event.get("result") == "made":
                    makes[team_id] = makes.get(team_id, 0) + 1
                else:
                    misses[team_id] = misses.get(team_id, 0) + 1

            attempts_per_frame.append(attempts.copy())
            makes_per_frame.append(makes.copy())
            misses_per_frame.append(misses.copy())

        return attempts_per_frame, makes_per_frame, misses_per_frame

    def _ball_is_near_hoop(self, ball_bbox, hoop_bbox):
        expanded_hoop = self._expand_bbox(hoop_bbox, x_scale=0.55, y_scale=0.9)
        overlap = calculate_overlap_ratio(ball_bbox, expanded_hoop)
        if overlap > 0:
            return True

        ball_center = get_center_of_bbox(ball_bbox)
        hoop_width = max(float(get_bbox_width(hoop_bbox)), 1.0)
        ball_width = max(float(get_bbox_width(ball_bbox)), 1.0)
        distance = point_to_bbox_distance(ball_center, expanded_hoop)
        return distance <= max(hoop_width * 0.65, ball_width * 1.5)

    def _ball_crossed_hoop(self, previous_ball_center, current_ball_center, hoop_bbox):
        hoop_center = get_center_of_bbox(hoop_bbox)
        hoop_width = max(float(get_bbox_width(hoop_bbox)), 1.0)
        hoop_height = max(float(get_bbox_height(hoop_bbox)), 1.0)
        score_window = self._expand_bbox(hoop_bbox, x_scale=0.18, y_scale=0.22)
        horizontal_margin = hoop_width * 0.75

        was_above = previous_ball_center[1] <= (hoop_center[1] - (hoop_height * 0.15))
        crossed_centerline = (
            previous_ball_center[1] <= hoop_center[1]
            and current_ball_center[1] >= hoop_center[1]
        )
        descending = current_ball_center[1] > previous_ball_center[1]
        stays_centered = (
            abs(previous_ball_center[0] - hoop_center[0]) <= horizontal_margin
            or abs(current_ball_center[0] - hoop_center[0]) <= horizontal_margin
        )
        entered_window = self._segment_intersects_bbox(
            previous_ball_center,
            current_ball_center,
            score_window,
        )

        return was_above and crossed_centerline and descending and stays_centered and entered_window

    def _expand_bbox(self, bbox, *, x_scale, y_scale):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        pad_x = width * x_scale
        pad_y = height * y_scale
        return [
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
        ]

    def _is_plausible_release(self, frame_players, ball_bbox, shooter_id):
        if ball_bbox is None or shooter_id == -1:
            return False

        player = frame_players.get(shooter_id)
        if player is None:
            return False

        player_bbox = player.get("bbox") or player.get("box")
        if player_bbox is None:
            return False

        ball_center = get_center_of_bbox(ball_bbox)
        player_center = get_center_of_bbox(player_bbox)
        player_height = max(float(get_bbox_height(player_bbox)), 1.0)
        player_width = max(float(get_bbox_width(player_bbox)), 1.0)
        shoulder_line = float(player_bbox[1]) + (player_height * 0.62)
        horizontal_gap = abs(float(ball_center[0]) - float(player_center[0]))
        distance_to_player = point_to_bbox_distance(ball_center, player_bbox)
        return (
            float(ball_center[1]) <= shoulder_line
            and (
                horizontal_gap <= (player_width * 0.95)
                or distance_to_player <= max(player_height * 0.55, player_width * 1.1)
            )
        )

    def _resolve_frame_count(self, frame_count, seconds):
        if frame_count is not None:
            return max(1, int(frame_count))
        return max(1, int(round(float(seconds) * max(self.fps, 1.0))))

    def _segment_intersects_bbox(self, point_a, point_b, bbox):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        ax, ay = [float(value) for value in point_a]
        bx, by = [float(value) for value in point_b]

        if x1 <= ax <= x2 and y1 <= ay <= y2:
            return True
        if x1 <= bx <= x2 and y1 <= by <= y2:
            return True

        edges = (
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        )
        for edge_start, edge_end in edges:
            if self._line_segments_intersect((ax, ay), (bx, by), edge_start, edge_end):
                return True
        return False

    def _line_segments_intersect(self, point_a, point_b, point_c, point_d):
        epsilon = 1e-6

        def orientation(point_one, point_two, point_three):
            return (
                (point_two[0] - point_one[0]) * (point_three[1] - point_one[1])
                - (point_two[1] - point_one[1]) * (point_three[0] - point_one[0])
            )

        def on_segment(point_one, point_two, point_three):
            return (
                min(point_one[0], point_three[0]) - epsilon
                <= point_two[0]
                <= max(point_one[0], point_three[0]) + epsilon
                and min(point_one[1], point_three[1]) - epsilon
                <= point_two[1]
                <= max(point_one[1], point_three[1]) + epsilon
            )

        def normalize_orientation(value):
            if abs(value) <= epsilon:
                return 0
            return 1 if value > 0 else -1

        orientation_a = orientation(point_a, point_b, point_c)
        orientation_b = orientation(point_a, point_b, point_d)
        orientation_c = orientation(point_c, point_d, point_a)
        orientation_d = orientation(point_c, point_d, point_b)

        normalized_a = normalize_orientation(orientation_a)
        normalized_b = normalize_orientation(orientation_b)
        normalized_c = normalize_orientation(orientation_c)
        normalized_d = normalize_orientation(orientation_d)

        if normalized_a == 0 and on_segment(point_a, point_c, point_b):
            return True
        if normalized_b == 0 and on_segment(point_a, point_d, point_b):
            return True
        if normalized_c == 0 and on_segment(point_c, point_a, point_d):
            return True
        if normalized_d == 0 and on_segment(point_c, point_b, point_d):
            return True

        return (
            normalized_a != normalized_b
            and normalized_c != normalized_d
        )

    def _get_release_position(self, player_positions_m, frame_num, shooter_id):
        if frame_num >= len(player_positions_m):
            return None
        return player_positions_m[frame_num].get(shooter_id)

    def _get_bbox(self, track_dict):
        if not track_dict:
            return None

        for track in track_dict.values():
            bbox = track.get("bbox") or track.get("box")
            if bbox is not None:
                return bbox

        return None
