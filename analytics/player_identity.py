from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from utils.player_id_utils import build_player_label


UNKNOWN_SCOPE = "U"
JERSEY_NUMBER_PATTERN = re.compile(r"\b(\d{1,2})\b")


class PlayerIdentityResolver:
    def __init__(
        self,
        *,
        max_samples_per_track=6,
        max_ocr_samples_per_track=2,
        min_crop_height_for_ocr=28,
        max_merge_gap_frames=180,
        min_merge_similarity=0.9,
    ):
        self.max_samples_per_track = max(1, int(max_samples_per_track))
        self.max_ocr_samples_per_track = max(1, int(max_ocr_samples_per_track))
        self.min_crop_height_for_ocr = max(1, int(min_crop_height_for_ocr))
        self.max_merge_gap_frames = max(1, int(max_merge_gap_frames))
        self.min_merge_similarity = float(min_merge_similarity)
        self._tesseract_path = shutil.which("tesseract")
        self._ocr_enabled = (
            self._tesseract_path is not None
            and os.environ.get("COURTVISION_ENABLE_JERSEY_OCR", "1").strip() not in {"0", "false", "False"}
        )

    def resolve(
        self,
        video_frames,
        player_tracks,
        team_assignments,
        *,
        mode="game",
        workout_player_id="",
    ):
        if not player_tracks:
            return {
                "player_tracks": player_tracks,
                "team_assignments": team_assignments,
                "identity_data": self._build_empty_identity_data(),
            }

        provisional_profiles = self._collect_profiles(
            video_frames,
            player_tracks,
            team_assignments,
        )
        clusters_by_scope = self._cluster_profiles(provisional_profiles)
        canonical_map, identity_entries = self._build_canonical_identities(
            clusters_by_scope,
            provisional_profiles,
            mode=mode,
            workout_player_id=workout_player_id,
        )
        resolved_tracks, resolved_assignments = self._remap_tracks(
            player_tracks,
            team_assignments,
            canonical_map,
            identity_entries,
        )

        return {
            "player_tracks": resolved_tracks,
            "team_assignments": resolved_assignments,
            "identity_data": self._build_identity_data(identity_entries),
        }

    def _collect_profiles(self, video_frames, player_tracks, team_assignments):
        profiles = {}

        for frame_num, frame_tracks in enumerate(player_tracks):
            frame_assignments = team_assignments[frame_num] if frame_num < len(team_assignments) else {}
            frame_height, frame_width = video_frames[frame_num].shape[:2]
            frame_center = (frame_width / 2.0, frame_height / 2.0)

            for player_label, player in frame_tracks.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue

                team_id = int(frame_assignments.get(player_label, player.get("team_id", -1)))
                profile = profiles.setdefault(
                    player_label,
                    {
                        "provisional_label": player_label,
                        "team_id": team_id,
                        "frames": [],
                        "frame_set": set(),
                        "samples": [],
                        "ocr_votes": Counter(),
                        "embeddings": [],
                        "tracked_frames": 0,
                        "best_area": 0.0,
                    },
                )

                profile["team_id"] = self._resolve_team_id(profile["team_id"], team_id)
                profile["frames"].append(frame_num)
                profile["frame_set"].add(frame_num)
                profile["tracked_frames"] += 1

                sample = self._build_sample(
                    video_frames[frame_num],
                    bbox,
                    frame_num=frame_num,
                    frame_center=frame_center,
                )
                if sample is None:
                    continue

                profile["best_area"] = max(profile["best_area"], sample["area"])
                profile["samples"].append(sample)

        for profile in profiles.values():
            profile["samples"].sort(
                key=lambda item: (-item["score"], item["frame_num"]),
            )
            profile["samples"] = profile["samples"][: self.max_samples_per_track]

            for sample_index, sample in enumerate(profile["samples"]):
                embedding = self._compute_appearance_embedding(sample["appearance_crop"])
                if embedding is not None:
                    profile["embeddings"].append(embedding)

                if sample_index < self.max_ocr_samples_per_track:
                    jersey_number = self._read_jersey_number(sample["ocr_crop"])
                    if jersey_number is not None:
                        profile["ocr_votes"][jersey_number] += 1

            if profile["embeddings"]:
                profile["embedding"] = np.mean(profile["embeddings"], axis=0)
                profile["embedding"] = _normalize_vector(profile["embedding"])
            else:
                profile["embedding"] = None

            profile["first_frame"] = min(profile["frames"])
            profile["last_frame"] = max(profile["frames"])
            profile["scope"] = self._scope_from_team_id(profile["team_id"])
            profile["jersey_number"], profile["jersey_confidence"] = self._resolve_jersey_number(
                profile["ocr_votes"]
            )

        return profiles

    def _build_sample(self, frame, bbox, *, frame_num, frame_center):
        full_crop = _crop_bbox(frame, bbox)
        if full_crop is None or full_crop.size == 0:
            return None

        appearance_crop = _crop_jersey_region(frame, bbox)
        if appearance_crop is None or appearance_crop.size == 0:
            appearance_crop = full_crop

        ocr_crop = _prepare_ocr_crop(appearance_crop)
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        area = max(float((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))), 0.0)
        center_distance = math.hypot(center_x - frame_center[0], center_y - frame_center[1])
        score = area - (center_distance * 12.0)

        return {
            "frame_num": int(frame_num),
            "bbox": [float(value) for value in bbox],
            "area": area,
            "score": float(score),
            "appearance_crop": appearance_crop,
            "ocr_crop": ocr_crop,
        }

    def _compute_appearance_embedding(self, crop):
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [12, 8, 4],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        return _normalize_vector(hist)

    def _read_jersey_number(self, crop):
        if not self._ocr_enabled or crop is None or crop.size == 0:
            return None
        if crop.shape[0] < self.min_crop_height_for_ocr:
            return None

        with tempfile.TemporaryDirectory(prefix="courtvision_ocr_") as temp_dir:
            image_path = Path(temp_dir) / "jersey.png"
            cv2.imwrite(str(image_path), crop)
            candidates = []
            for psm in ("7", "8"):
                candidate = _run_tesseract_digits(self._tesseract_path, image_path, psm=psm)
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            return None

        number, _ = Counter(candidates).most_common(1)[0]
        return number

    def _resolve_jersey_number(self, votes):
        if not votes:
            return None, 0.0

        jersey_number, count = votes.most_common(1)[0]
        total_votes = sum(votes.values())
        confidence = count / max(total_votes, 1)
        if total_votes > 1 and confidence < 0.6:
            return None, float(confidence)
        return jersey_number, float(confidence)

    def _cluster_profiles(self, profiles):
        labels_by_scope = defaultdict(list)
        for profile in profiles.values():
            labels_by_scope[profile["scope"]].append(profile["provisional_label"])

        clusters_by_scope = {}
        for scope, labels in labels_by_scope.items():
            clusters = [
                {
                    "members": [label],
                    "frame_set": set(profiles[label]["frame_set"]),
                    "first_frame": profiles[label]["first_frame"],
                    "last_frame": profiles[label]["last_frame"],
                    "embedding": profiles[label]["embedding"],
                    "team_id": profiles[label]["team_id"],
                    "jersey_number": profiles[label]["jersey_number"],
                    "tracked_frames": profiles[label]["tracked_frames"],
                }
                for label in sorted(labels, key=lambda value: profiles[value]["first_frame"])
            ]

            merged = True
            while merged:
                merged = False
                best_pair = None
                best_score = -1.0

                for left_index in range(len(clusters)):
                    for right_index in range(left_index + 1, len(clusters)):
                        score = self._score_merge_candidate(
                            clusters[left_index],
                            clusters[right_index],
                        )
                        if score is None or score <= best_score:
                            continue
                        best_pair = (left_index, right_index)
                        best_score = score

                if best_pair is None:
                    continue

                left_index, right_index = best_pair
                clusters[left_index] = self._merge_clusters(
                    clusters[left_index],
                    clusters[right_index],
                )
                del clusters[right_index]
                merged = True

            clusters_by_scope[scope] = clusters

        return clusters_by_scope

    def _score_merge_candidate(self, left_cluster, right_cluster):
        if left_cluster["frame_set"] & right_cluster["frame_set"]:
            return None

        gap_frames = _frame_gap(left_cluster, right_cluster)
        left_number = left_cluster.get("jersey_number")
        right_number = right_cluster.get("jersey_number")

        if left_number is not None and right_number is not None and left_number == right_number:
            return 2.0

        if gap_frames > self.max_merge_gap_frames:
            return None

        similarity = _embedding_similarity(
            left_cluster.get("embedding"),
            right_cluster.get("embedding"),
        )
        if similarity < self.min_merge_similarity:
            return None

        gap_penalty = min(gap_frames / self.max_merge_gap_frames, 1.0) * 0.08
        return similarity - gap_penalty

    def _merge_clusters(self, left_cluster, right_cluster):
        tracked_frames = left_cluster["tracked_frames"] + right_cluster["tracked_frames"]
        embedding = _weighted_embedding_mean(
            left_cluster.get("embedding"),
            left_cluster["tracked_frames"],
            right_cluster.get("embedding"),
            right_cluster["tracked_frames"],
        )
        jersey_number = left_cluster.get("jersey_number")
        if jersey_number is None:
            jersey_number = right_cluster.get("jersey_number")

        return {
            "members": left_cluster["members"] + right_cluster["members"],
            "frame_set": left_cluster["frame_set"] | right_cluster["frame_set"],
            "first_frame": min(left_cluster["first_frame"], right_cluster["first_frame"]),
            "last_frame": max(left_cluster["last_frame"], right_cluster["last_frame"]),
            "embedding": embedding,
            "team_id": self._resolve_team_id(left_cluster["team_id"], right_cluster["team_id"]),
            "jersey_number": jersey_number,
            "tracked_frames": tracked_frames,
        }

    def _build_canonical_identities(
        self,
        clusters_by_scope,
        profiles,
        *,
        mode,
        workout_player_id,
    ):
        canonical_map = {}
        identity_entries = {}
        workout_player_id = (workout_player_id or "").strip()
        primary_workout_identity = None
        primary_workout_frames = -1

        for scope in sorted(clusters_by_scope, key=_scope_sort_key):
            clusters = sorted(clusters_by_scope[scope], key=lambda item: item["first_frame"])
            for index, cluster in enumerate(clusters, start=1):
                canonical_id = build_player_label(scope, index)
                track_count = cluster["tracked_frames"]
                jersey_number = cluster.get("jersey_number")
                display_id = jersey_number if jersey_number is not None else canonical_id
                display_name = (
                    f"{canonical_id} (#{jersey_number})"
                    if jersey_number is not None
                    else canonical_id
                )
                identity_source = "team_slot"
                if jersey_number is not None:
                    identity_source = "jersey_ocr"

                identity_entry = {
                    "player_id": canonical_id,
                    "display_id": display_id,
                    "display_name": display_name,
                    "team_id": int(cluster["team_id"]) if cluster["team_id"] != -1 else -1,
                    "jersey_number": jersey_number,
                    "identity_source": identity_source,
                    "tracked_frames": int(track_count),
                    "provisional_labels": list(cluster["members"]),
                    "first_frame": int(cluster["first_frame"]),
                    "last_frame": int(cluster["last_frame"]),
                }

                for provisional_label in cluster["members"]:
                    canonical_map[provisional_label] = canonical_id

                identity_entries[canonical_id] = identity_entry

                if mode == "workout" and workout_player_id and track_count > primary_workout_frames:
                    primary_workout_identity = canonical_id
                    primary_workout_frames = track_count

        if primary_workout_identity is not None:
            identity_entry = identity_entries[primary_workout_identity]
            identity_entry["display_id"] = workout_player_id
            identity_entry["display_name"] = f"Workout #{workout_player_id}"
            identity_entry["identity_source"] = "manual_workout"
            identity_entry["workout_player_id"] = workout_player_id

        return canonical_map, identity_entries

    def _remap_tracks(
        self,
        player_tracks,
        team_assignments,
        canonical_map,
        identity_entries,
    ):
        resolved_tracks = []
        resolved_assignments = []

        for frame_num, frame_tracks in enumerate(player_tracks):
            frame_assignments = team_assignments[frame_num] if frame_num < len(team_assignments) else {}
            resolved_frame_tracks = {}
            resolved_frame_assignments = {}

            for provisional_label, player in frame_tracks.items():
                canonical_id = canonical_map.get(provisional_label, provisional_label)
                identity_entry = identity_entries.get(canonical_id)
                player_copy = dict(player)
                player_copy["provisional_player_label"] = provisional_label
                player_copy["player_label"] = canonical_id
                player_copy["canonical_player_id"] = canonical_id

                if identity_entry is not None:
                    player_copy["display_id"] = identity_entry["display_id"]
                    player_copy["display_name"] = identity_entry["display_name"]
                    player_copy["jersey_number"] = identity_entry["jersey_number"]
                    player_copy["identity_source"] = identity_entry["identity_source"]
                    player_copy["team_id"] = identity_entry["team_id"]

                existing_player = resolved_frame_tracks.get(canonical_id)
                if existing_player is not None:
                    existing_bbox = existing_player.get("bbox") or existing_player.get("box")
                    current_bbox = player_copy.get("bbox") or player_copy.get("box")
                    if _bbox_area(current_bbox) <= _bbox_area(existing_bbox):
                        continue

                resolved_frame_tracks[canonical_id] = player_copy
                resolved_frame_assignments[canonical_id] = int(frame_assignments.get(provisional_label, player_copy.get("team_id", -1)))

            resolved_tracks.append(resolved_frame_tracks)
            resolved_assignments.append(resolved_frame_assignments)

        return resolved_tracks, resolved_assignments

    def _build_identity_data(self, identity_entries):
        players = sorted(
            identity_entries.values(),
            key=lambda item: (_scope_sort_key(self._scope_from_team_id(item["team_id"])), item["player_id"]),
        )
        players_with_numbers = sum(1 for item in players if item.get("jersey_number"))
        primary_identity = next(
            (
                item["player_id"]
                for item in players
                if item.get("identity_source") == "manual_workout"
            ),
            None,
        )

        return {
            "appearance_backend": "color_histogram",
            "ocr_backend": "tesseract" if self._ocr_enabled else "none",
            "resolved_players": len(players),
            "players_with_numbers": int(players_with_numbers),
            "primary_identity": primary_identity,
            "players": players,
            "players_by_id": {item["player_id"]: item for item in players},
        }

    def _build_empty_identity_data(self):
        return {
            "appearance_backend": "color_histogram",
            "ocr_backend": "tesseract" if self._ocr_enabled else "none",
            "resolved_players": 0,
            "players_with_numbers": 0,
            "primary_identity": None,
            "players": [],
            "players_by_id": {},
        }

    def _resolve_team_id(self, previous_team_id, current_team_id):
        if current_team_id in (1, 2):
            return int(current_team_id)
        if previous_team_id in (1, 2):
            return int(previous_team_id)
        return -1

    def _scope_from_team_id(self, team_id):
        if int(team_id) in (1, 2):
            return int(team_id)
        return UNKNOWN_SCOPE


def _crop_bbox(frame, bbox):
    x1 = max(0, int(math.floor(float(bbox[0]))))
    y1 = max(0, int(math.floor(float(bbox[1]))))
    x2 = min(frame.shape[1], int(math.ceil(float(bbox[2]))))
    y2 = min(frame.shape[0], int(math.ceil(float(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _crop_jersey_region(frame, bbox):
    player_crop = _crop_bbox(frame, bbox)
    if player_crop is None or player_crop.size == 0:
        return None

    height, width = player_crop.shape[:2]
    x1 = int(width * 0.2)
    x2 = int(width * 0.8)
    y1 = int(height * 0.14)
    y2 = int(height * 0.62)
    if x2 <= x1 or y2 <= y1:
        return player_crop
    return player_crop[y1:y2, x1:x2]


def _prepare_ocr_crop(crop):
    if crop is None or crop.size == 0:
        return None

    enlarged = cv2.resize(
        crop,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def _run_tesseract_digits(tesseract_path, image_path, *, psm):
    try:
        process = subprocess.run(
            [
                tesseract_path,
                str(image_path),
                "stdout",
                "--psm",
                str(psm),
                "--oem",
                "1",
                "-c",
                "tessedit_char_whitelist=0123456789",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = JERSEY_NUMBER_PATTERN.search(process.stdout or "")
    if match is None:
        return None

    number = match.group(1).lstrip("0") or "0"
    if not number.isdigit():
        return None
    value = int(number)
    if value > 99:
        return None
    return f"{value}"


def _normalize_vector(vector):
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def _embedding_similarity(left_embedding, right_embedding):
    if left_embedding is None or right_embedding is None:
        return 0.0
    return float(np.dot(left_embedding, right_embedding))


def _weighted_embedding_mean(left_embedding, left_weight, right_embedding, right_weight):
    if left_embedding is None:
        return right_embedding
    if right_embedding is None:
        return left_embedding

    total_weight = max(float(left_weight) + float(right_weight), 1.0)
    combined = (
        (left_embedding * float(left_weight))
        + (right_embedding * float(right_weight))
    ) / total_weight
    return _normalize_vector(combined)


def _frame_gap(left_cluster, right_cluster):
    if left_cluster["last_frame"] < right_cluster["first_frame"]:
        return right_cluster["first_frame"] - left_cluster["last_frame"]
    if right_cluster["last_frame"] < left_cluster["first_frame"]:
        return left_cluster["first_frame"] - right_cluster["last_frame"]
    return 0


def _bbox_area(bbox):
    if bbox is None:
        return 0.0
    return max((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])), 0.0)


def _scope_sort_key(scope):
    if scope == UNKNOWN_SCOPE:
        return 99
    return int(scope)
