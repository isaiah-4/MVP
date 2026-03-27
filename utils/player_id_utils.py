from __future__ import annotations

from collections import Counter, defaultdict
from math import hypot
import re

from .bbox_utils import calculate_bbox_area, get_bbox_height, get_center_of_bbox


TEAM_SCOPES = (1, 2)
UNKNOWN_SCOPE = "U"
PLAYER_LABEL_PATTERN = re.compile(r"^T(?P<team>\d+)-(?P<slot>\d+)$")
UNKNOWN_LABEL_PATTERN = re.compile(r"^U-(?P<slot>\d+)$")


def normalize_player_track_ids_by_team(
    player_tracks,
    team_assignments,
    *,
    team_colors=None,
    max_players_per_team=5,
    max_unknown_players=2,
    max_inactive_gap=45,
    max_match_distance=140.0,
    min_iou_for_match=0.05,
):
    max_players_per_team = max(1, int(max_players_per_team))
    max_unknown_players = max(1, int(max_unknown_players))
    max_inactive_gap = max(1, int(max_inactive_gap))
    max_match_distance = float(max_match_distance)
    min_iou_for_match = float(min_iou_for_match)

    primary_team_by_raw_track = _build_primary_team_map(team_assignments)
    slot_states_by_scope = {
        1: {},
        2: {},
        UNKNOWN_SCOPE: {},
    }
    raw_to_label = {}
    normalized_tracks = []
    normalized_team_assignments = []

    for frame_num, frame_tracks in enumerate(player_tracks):
        frame_assignment = (
            team_assignments[frame_num]
            if frame_num < len(team_assignments)
            else {}
        )
        scoped_candidates = {
            1: [],
            2: [],
            UNKNOWN_SCOPE: [],
        }

        for raw_track_id, player in frame_tracks.items():
            bbox = player.get("bbox") or player.get("box")
            if bbox is None:
                continue

            resolved_team_id = frame_assignment.get(
                raw_track_id,
                primary_team_by_raw_track.get(raw_track_id, -1),
            )
            scope = resolved_team_id if resolved_team_id in TEAM_SCOPES else UNKNOWN_SCOPE
            scoped_candidates[scope].append(
                {
                    "raw_track_id": raw_track_id,
                    "player": player,
                    "bbox": bbox,
                    "team_id": resolved_team_id,
                    "area": float(calculate_bbox_area(bbox)),
                }
            )

        normalized_frame_tracks = {}
        normalized_frame_assignment = {}

        for scope, max_slots in (
            (1, max_players_per_team),
            (2, max_players_per_team),
            (UNKNOWN_SCOPE, max_unknown_players),
        ):
            candidates = sorted(
                scoped_candidates[scope],
                key=lambda item: (-item["area"], str(item["raw_track_id"])),
            )[:max_slots]
            used_labels = set()
            unassigned_candidates = []

            for candidate in candidates:
                raw_track_id = candidate["raw_track_id"]
                label = raw_to_label.get(raw_track_id)
                if label is None or label in used_labels or _get_label_scope(label) != scope:
                    unassigned_candidates.append(candidate)
                    continue

                slot_state = slot_states_by_scope[scope].get(label)
                if slot_state is None or slot_state.get("raw_track_id") != raw_track_id:
                    unassigned_candidates.append(candidate)
                    continue

                _assign_player_label(
                    normalized_frame_tracks,
                    normalized_frame_assignment,
                    used_labels,
                    slot_states_by_scope[scope],
                    raw_to_label,
                    frame_num,
                    candidate,
                    label,
                    team_colors=team_colors,
                )

            remaining_candidates = []
            for candidate in unassigned_candidates:
                label = _match_inactive_player_label(
                    candidate["bbox"],
                    frame_num,
                    slot_states_by_scope[scope],
                    used_labels,
                    max_inactive_gap=max_inactive_gap,
                    max_match_distance=max_match_distance,
                    min_iou_for_match=min_iou_for_match,
                )
                if label is None:
                    remaining_candidates.append(candidate)
                    continue

                _assign_player_label(
                    normalized_frame_tracks,
                    normalized_frame_assignment,
                    used_labels,
                    slot_states_by_scope[scope],
                    raw_to_label,
                    frame_num,
                    candidate,
                    label,
                    team_colors=team_colors,
                )

            for candidate in remaining_candidates:
                raw_track_id = candidate["raw_track_id"]
                existing_label = raw_to_label.get(raw_track_id)
                if existing_label is not None and existing_label in used_labels:
                    continue

                available_labels = [
                    build_player_label(scope, slot_index)
                    for slot_index in range(1, max_slots + 1)
                    if build_player_label(scope, slot_index) not in used_labels
                ]
                if not available_labels:
                    break

                _assign_player_label(
                    normalized_frame_tracks,
                    normalized_frame_assignment,
                    used_labels,
                    slot_states_by_scope[scope],
                    raw_to_label,
                    frame_num,
                    candidate,
                    available_labels[0],
                    team_colors=team_colors,
                )

        normalized_tracks.append(normalized_frame_tracks)
        normalized_team_assignments.append(normalized_frame_assignment)

    return normalized_tracks, normalized_team_assignments


def build_player_label(scope, slot_index):
    slot_index = int(slot_index)
    if scope == UNKNOWN_SCOPE:
        return f"U-{slot_index}"
    return f"T{int(scope)}-{slot_index}"


def sort_player_identifier(value):
    if value in (None, "", -1):
        return (99, 99, "")

    if isinstance(value, str):
        team_match = PLAYER_LABEL_PATTERN.fullmatch(value)
        if team_match is not None:
            return (
                int(team_match.group("team")),
                int(team_match.group("slot")),
                value,
            )

        unknown_match = UNKNOWN_LABEL_PATTERN.fullmatch(value)
        if unknown_match is not None:
            return (
                98,
                int(unknown_match.group("slot")),
                value,
            )

    return (97, 0, str(value))


def _assign_player_label(
    normalized_frame_tracks,
    normalized_frame_assignment,
    used_labels,
    slot_states,
    raw_to_label,
    frame_num,
    candidate,
    label,
    team_colors=None,
):
    raw_track_id = candidate["raw_track_id"]
    bbox = candidate["bbox"]
    team_id = candidate["team_id"]
    player = dict(candidate["player"])
    player["display_id"] = label
    player["player_label"] = label
    player["raw_track_id"] = raw_track_id
    if team_id in TEAM_SCOPES:
        player["team_id"] = int(team_id)
        if team_colors is not None:
            player["team_color"] = team_colors.get(int(team_id), player.get("team_color"))

    normalized_frame_tracks[label] = player
    normalized_frame_assignment[label] = int(team_id) if team_id in TEAM_SCOPES else -1
    used_labels.add(label)
    raw_to_label[raw_track_id] = label
    slot_states[label] = {
        "raw_track_id": raw_track_id,
        "bbox": [float(value) for value in bbox],
        "last_seen_frame": int(frame_num),
    }


def _build_primary_team_map(team_assignments):
    votes_by_raw_track = defaultdict(Counter)

    for frame_assignment in team_assignments:
        for raw_track_id, team_id in frame_assignment.items():
            if team_id in TEAM_SCOPES:
                votes_by_raw_track[raw_track_id][int(team_id)] += 1

    return {
        raw_track_id: votes.most_common(1)[0][0]
        for raw_track_id, votes in votes_by_raw_track.items()
        if votes
    }


def _get_label_scope(label):
    if isinstance(label, str):
        team_match = PLAYER_LABEL_PATTERN.fullmatch(label)
        if team_match is not None:
            return int(team_match.group("team"))
        if UNKNOWN_LABEL_PATTERN.fullmatch(label) is not None:
            return UNKNOWN_SCOPE
    return None


def _match_inactive_player_label(
    bbox,
    frame_num,
    slot_states,
    used_labels,
    *,
    max_inactive_gap,
    max_match_distance,
    min_iou_for_match,
):
    best_label = None
    best_score = None
    current_height = max(float(get_bbox_height(bbox)), 1.0)

    for label, slot_state in slot_states.items():
        if label in used_labels:
            continue

        frame_gap = int(frame_num) - int(slot_state["last_seen_frame"])
        if frame_gap <= 0 or frame_gap > max_inactive_gap:
            continue

        previous_bbox = slot_state["bbox"]
        iou = _calculate_iou(bbox, previous_bbox)
        center_distance = _calculate_center_distance(bbox, previous_bbox)
        previous_height = max(float(get_bbox_height(previous_bbox)), 1.0)
        distance_limit = max(
            max_match_distance,
            ((current_height + previous_height) * 0.6) * min(frame_gap + 1, 4),
        )

        if iou < min_iou_for_match and center_distance > distance_limit:
            continue

        score = (-float(iou), float(center_distance), int(frame_gap), str(label))
        if best_score is None or score < best_score:
            best_score = score
            best_label = label

    return best_label


def _calculate_center_distance(bbox_a, bbox_b):
    center_a = get_center_of_bbox(bbox_a)
    center_b = get_center_of_bbox(bbox_b)
    return float(hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))


def _calculate_iou(bbox_a, bbox_b):
    x_left = max(float(bbox_a[0]), float(bbox_b[0]))
    y_top = max(float(bbox_a[1]), float(bbox_b[1]))
    x_right = min(float(bbox_a[2]), float(bbox_b[2]))
    y_bottom = min(float(bbox_a[3]), float(bbox_b[3]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = calculate_bbox_area(bbox_a)
    area_b = calculate_bbox_area(bbox_b)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)
