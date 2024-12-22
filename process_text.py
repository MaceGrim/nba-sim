import csv
import glob
import os
import re
from tqdm import tqdm

def determine_period_label(period_num):
    if period_num == 1:
        return "Q1"
    elif period_num == 2:
        return "Q2"
    elif period_num == 3:
        return "Q3"
    elif period_num == 4:
        return "Q4"
    else:
        return f"OT{period_num-4}"

def determine_shot_type(distance):
    if distance is None:
        return "2PT"
    return "3PT" if distance > 22 else "2PT"

def underscore_name(name):
    if not name:
        return name
    name = re.sub(r"[^\w]", "_", name.strip())
    return name

def parse_teams_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('-')
    last_part = parts[-1].replace('.csv', '')
    if '@' in last_part:
        away_team, home_team = last_part.split('@', 1)
        return away_team, home_team
    return "AWAY", "HOME"

def process_file(filepath, output_file):
    away_team, home_team = parse_teams_from_filename(filepath)

    all_rows = []
    away_team_players = set()
    home_team_players = set()

    # Read rows and identify players
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            for player in [row['a1'], row['a2'], row['a3'], row['a4'], row['a5']]:
                if player: away_team_players.add(player.strip())
            for player in [row['h1'], row['h2'], row['h3'], row['h4'], row['h5']]:
                if player: home_team_players.add(player.strip())

    on_court_away = set([underscore_name(away_team_player) for away_team_player in away_team_players])
    on_court_home = set([underscore_name(home_team_player) for home_team_player in home_team_players])
    possession = "Home"  # Start with Home team by default for jump ball
    last_shot_team = None

    # Write start marker
    output_file.write("<START_GAME>\n")

    # Write Home and Away team names
    output_file.write(f"HomeTeam {home_team}\n")
    output_file.write(f"AwayTeam {away_team}\n\n")

    for i, row in enumerate(all_rows):
        # Event data
        event_type = row['event_type'].lower() if row['event_type'] else ""
        event_type_long = row["type"].lower() if row["type"] else ""
        team = row['team'].strip() if row['team'] else ""
        player = underscore_name(row['player']) if row['player'] else "NoPlayer"
        possession_team = row['possession'].strip() if 'possession' in row and row['possession'] else None

        # Handle jump ball
        if event_type == "jump ball":
            tip_to_player = possession_team if possession_team else "Unknown"
            possession = "Home" if underscore_name(tip_to_player) in on_court_home else "Away"
            event_tokens = ["GameAdmin", "JumpBall", f"TipTo:{underscore_name(tip_to_player)}", "<EVENT_END>"]

        # Shots
        elif event_type == "shot":
            outcome = "Made" if row['result'].lower() == "made" else "Missed"

            shot_distance = row['shot_distance'] if row['shot_distance'] else "UnknownDist"
            shot_location_x = row['original_x'] if row['original_x'] else "UnknownX"
            shot_location_y = row['original_y'] if row['original_y'] else "UnknownY"

            assister = underscore_name(row['assist']) if row['assist'] else "NoAssist"
            blocker = underscore_name(row['block']) if row['block'] else "NoBlock"

            shot_type = determine_shot_type(float(shot_distance)) if shot_distance != "UnknownDist" else "Unknown"
            event_tokens = [underscore_name(team), player, "Shot", shot_type, shot_location_x, shot_location_y, outcome, assister, blocker, "<EVENT_END>"]
            last_shot_team = "Home" if player in on_court_home else "Away"
            if outcome == "Made":  # Switch possession if shot is made
                possession = "Away" if possession == "Home" else "Home"

        # Rebounds
        elif event_type == "rebound":
            rebound_type = "Offensive" if "offensive" in event_type_long else "Defensive"
            event_tokens = [underscore_name(team), player, "Rebound", f"{rebound_type}Rebound", "<EVENT_END>"]
            if player in on_court_home:
                possession = "Home"
            elif player in on_court_away:
                possession = "Away"
            
            if player == "NoPlayer":
                if rebound_type == "Defensive":
                    possession = "Home" if last_shot_team == "Away" else "Home"
                

        # Turnovers
        elif event_type == "turnover":
            event_tokens = [underscore_name(team), player, "Turnover", "<EVENT_END>"]
            possession = "Away" if possession == "Home" else "Home"

        elif event_type == "substitution":
            in_player = underscore_name(row['entered'])
            out_player = underscore_name(row['left'])

            # Update on-court players
            if in_player in away_team_players:
                on_court_away.add(in_player)
                on_court_away.remove(out_player)
            elif in_player in home_team_players:
                on_court_home.add(in_player)
                on_court_home.remove(out_player)

            event_tokens = [underscore_name(team), out_player, "SubOut", in_player, "SubIn", "<EVENT_END>"]

        elif event_type == "free throw":
            outcome = "Made" if row['result'].lower() == "made" else "Missed"
            num = row['num']
            outof = row['outof']

            event_tokens = [underscore_name(team), player, "FreeThrow", num, outof, outcome, "<EVENT_END>"]

            if num == outof:
                last_shot_team = "Home" if player in on_court_home else "Away"
                if outcome == "Made":
                    possession = "Away" if possession == "Home" else "Away"

        elif event_type == "foul":
            fouled_player = underscore_name(row['opponent'])
            foul_type = row['reason']
            event_tokens = [underscore_name(team), player, "Foul", fouled_player, foul_type, "<EVENT_END>"]

        # Default for unknown events
        else:
            event_tokens = ["GameAdmin", underscore_name(event_type.title()), "<EVENT_END>"]

        # Print event
        output_file.write(" ".join(event_tokens) + "\n")

        output_file.write("\n")

        # Update state_t+1
        period_label = determine_period_label(int(row['period']))
        context_tokens = [
            period_label,
            row['remaining_time'],
            row['away_score'],
            row['home_score'],
            possession,
            underscore_name(away_team), "OnCourtAway", *sorted(map(underscore_name, on_court_away)),
            underscore_name(home_team), "OnCourtHome", *sorted(map(underscore_name, on_court_home))
        ]
        output_file.write(" ".join(context_tokens) + "\n\n")

    # Write end marker
    output_file.write("<END_GAME>\n\n")

def main():
    input_pattern = "./data/*.csv"
    output_filename = "all_games.txt"

    with open(output_filename, "w", encoding="utf-8") as out_f:
        for filepath in tqdm(glob.glob(input_pattern)):
            process_file(filepath, out_f)

if __name__ == "__main__":
    main()
