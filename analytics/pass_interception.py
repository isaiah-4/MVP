from collections import defaultdict


class PassInterceptionDetector:
    def detect(self, possession_data):
        team_pass_counts = {
            1: 0,
            2: 0,
        }
        team_interception_counts = {
            1: 0,
            2: 0,
        }
        passes_per_frame = []
        interceptions_per_frame = []
        events = []

        last_valid_holder = -1
        last_valid_team = -1

        for frame_num, current_holder in enumerate(possession_data["player"]):
            current_team = possession_data["team"][frame_num]

            if current_holder != -1:
                if last_valid_holder != -1 and current_holder != last_valid_holder:
                    if current_team != -1 and last_valid_team != -1:
                        if current_team == last_valid_team:
                            team_pass_counts[current_team] += 1
                            events.append(
                                {
                                    "frame_num": frame_num,
                                    "type": "pass",
                                    "from_player": last_valid_holder,
                                    "to_player": current_holder,
                                    "team_id": current_team,
                                }
                            )
                        else:
                            team_interception_counts[current_team] += 1
                            events.append(
                                {
                                    "frame_num": frame_num,
                                    "type": "interception",
                                    "from_player": last_valid_holder,
                                    "to_player": current_holder,
                                    "team_id": current_team,
                                }
                            )

                last_valid_holder = current_holder
                last_valid_team = current_team

            passes_per_frame.append(team_pass_counts.copy())
            interceptions_per_frame.append(team_interception_counts.copy())

        events_by_frame = defaultdict(list)
        for event in events:
            events_by_frame[event["frame_num"]].append(event)

        return {
            "events": events,
            "events_by_frame": dict(events_by_frame),
            "passes_per_frame": passes_per_frame,
            "interceptions_per_frame": interceptions_per_frame,
        }
