from collections import Counter, defaultdict

import cv2
import numpy as np


class TeamAssigner:
    def __init__(self, sample_frames=40):
        self.sample_frames = sample_frames
        self.team_colors = {
            1: (255, 0, 0),
            2: (0, 0, 255),
        }
        self.team_centers = {}
        self.track_to_team = {}

    def assign_teams(self, video_frames, player_tracks):
        samples = []
        sample_track_ids = []
        max_frames = min(len(video_frames), self.sample_frames)

        for frame_num in range(max_frames):
            frame = video_frames[frame_num]
            for track_id, player in player_tracks[frame_num].items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue

                color_feature = self._extract_player_color(frame, bbox)
                if color_feature is None:
                    continue

                samples.append(color_feature)
                sample_track_ids.append(track_id)

        if len(samples) >= 2:
            labels, centers = self._cluster_color_samples(samples)
            cluster_to_team = self._build_cluster_to_team(centers)
            track_votes = defaultdict(list)

            for label, track_id in zip(labels.flatten().tolist(), sample_track_ids):
                track_votes[track_id].append(cluster_to_team[int(label)])

            self.team_centers = {
                cluster_to_team[index]: centers[index]
                for index in range(len(centers))
            }
            self.team_colors = {
                team_id: self._build_display_color(center)
                for team_id, center in self.team_centers.items()
            }
            self.track_to_team = {
                track_id: Counter(votes).most_common(1)[0][0]
                for track_id, votes in track_votes.items()
            }

        assignments = []
        for frame_num, frame_tracks in enumerate(player_tracks):
            frame_assignment = {}
            frame = video_frames[frame_num]

            for track_id, player in frame_tracks.items():
                bbox = player.get("bbox") or player.get("box")
                if bbox is None:
                    continue

                team_id = self.track_to_team.get(track_id)
                if team_id is None:
                    color_feature = self._extract_player_color(frame, bbox)
                    team_id = self.predict_team(color_feature)
                    if team_id is not None:
                        self.track_to_team[track_id] = team_id

                if team_id is None:
                    continue

                team_color = self.get_team_color(team_id)
                frame_assignment[track_id] = team_id
                player["team_id"] = team_id
                player["team_color"] = team_color

            assignments.append(frame_assignment)

        return assignments

    def get_team_color(self, team_id):
        return self.team_colors.get(team_id, (0, 0, 255))

    def predict_team(self, color_feature):
        if color_feature is None or not self.team_centers:
            return None

        best_team_id = None
        best_distance = float("inf")
        color_feature = np.asarray(color_feature, dtype=np.float32)

        for team_id, center in self.team_centers.items():
            distance = float(np.linalg.norm(color_feature - center))
            if distance < best_distance:
                best_distance = distance
                best_team_id = team_id

        return best_team_id

    def _cluster_color_samples(self, samples):
        sample_array = np.asarray(samples, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            0.2,
        )
        _, labels, centers = cv2.kmeans(
            sample_array,
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        return labels, centers

    def _build_cluster_to_team(self, centers):
        center_sums = [float(np.sum(center)) for center in centers]
        ordered_indices = np.argsort(center_sums)
        return {
            int(ordered_indices[0]): 1,
            int(ordered_indices[1]): 2,
        }

    def _build_display_color(self, center):
        color = np.clip(center.astype(int), 35, 255)
        if int(np.sum(color)) > 680:
            color = np.clip(color - 55, 35, 255)
        return tuple(int(value) for value in color.tolist())

    def _extract_player_color(self, frame, bbox):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(frame_width - 1, x1))
        x2 = max(0, min(frame_width, x2))
        y1 = max(0, min(frame_height - 1, y1))
        y2 = max(0, min(frame_height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        width = x2 - x1
        height = y2 - y1
        crop_x1 = x1 + int(width * 0.2)
        crop_x2 = x2 - int(width * 0.2)
        crop_y1 = y1 + int(height * 0.08)
        crop_y2 = y1 + int(height * 0.55)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3)
        return np.median(pixels, axis=0).astype(np.float32)
