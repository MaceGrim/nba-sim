
import pandas as pd
from glob import glob
from tqdm import tqdm


# Generates tokens for all games


# Extract home and away teams from the file name
def get_teams_from_filename(filename):
    teams_info = filename.split('-')[-1].split('.')[0] # Extracts 'AWAY_TEAM@HOME_TEAM'
    away_team, home_team = teams_info.split('@')
    return home_team, away_team


def generate_individual_context_tokens(df, row):
    """
    Generates individual context setting tokens based on the starting line-up and bench players.

    Parameters:
    - df: The game data dataframe.
    - row: The first row of the game data.

    Returns:
    - context_tokens: A list of tokens setting the context for the game.
    """
    away_starting = row[['a1', 'a2', 'a3', 'a4', 'a5']].tolist()
    home_starting = row[['h1', 'h2', 'h3', 'h4', 'h5']].tolist()

    # For simplicity, we'll assume the rest of the unique players in the game data (beyond the starting five)
    # are the bench players.
    all_away_players = pd.concat([df['a1'], df['a2'], df['a3'], df['a4'], df['a5']]).unique().tolist()
    all_home_players = pd.concat([df['h1'], df['h2'], df['h3'], df['h4'], df['h5']]).unique().tolist()

    away_bench = [player for player in all_away_players if player not in away_starting]
    home_bench = [player for player in all_home_players if player not in home_starting]

    # Construct the context tokens with individual tokens for each player
    context_tokens = []
    context_tokens.extend([f"AWAY_START|{player}" for player in away_starting])
    context_tokens.extend([f"AWAY_BENCH|{player}" for player in away_bench])
    context_tokens.extend([f"HOME_START|{player}" for player in home_starting])
    context_tokens.extend([f"HOME_BENCH|{player}" for player in home_bench])

    return context_tokens


def generate_tokens(df, home_team, away_team):
    tokens = ["START_OF_GAME"]
    prev_active_players_home = []
    prev_active_players_away = []

    # Add context setting tokens
    tokens.extend(generate_individual_context_tokens(df, df.iloc[0]))

    for idx, row in df.iterrows():
        home_team_players = row[['h1', 'h2', 'h3', 'h4', 'h5']].tolist()
        away_team_players = row[['a1', 'a2', 'a3', 'a4', 'a5']].tolist()

        # Check for substitutions
        if prev_active_players_home:
            for player in prev_active_players_home:
                if player not in home_team_players:
                    sub_out = player
                    for new_player in home_team_players:
                        if new_player not in prev_active_players_home:
                            sub_in = new_player
                            tokens.append(f"SUB|{sub_out}|{sub_in}|HOME")
                            break

        if prev_active_players_away:
            for player in prev_active_players_away:
                if player not in away_team_players:
                    sub_out = player
                    for new_player in away_team_players:
                        if new_player not in prev_active_players_away:
                            sub_in = new_player
                            tokens.append(f"SUB|{sub_out}|{sub_in}|AWAY") # TODO: This data is in the data maybe
                            break

        # Generate play token
        player = row['player']
        event_type = row['event_type'] if "3pt" not in str(row["type"]) else "3pt Shot"
        result = row['result'] if pd.notna(row['result']) else ""
        team_flag = "HOME" if row['team'] == home_team else "AWAY"
        tokens.append(f"{player}|{event_type}|{result}|{team_flag}")

        # Set previous active players for next iteration
        prev_active_players_home = home_team_players
        prev_active_players_away = away_team_players

    # Add End of Game token
    tokens.append("END_OF_GAME")

    return tokens


if __name__ == "__main__":

    for game_file in tqdm(glob('./data/*.csv')):

        print(game_file)

        # Extract home and away teams from filenames and generate tokens
        game_data = pd.read_csv(game_file)
        home_team, away_team = get_teams_from_filename(game_file)
        game_tokens = generate_tokens(game_data, home_team, away_team)

        output_name = game_file.split('\\')[-1].split('.')[0]

        # Save the tokens to files for further processing
        with open(f'./token_files/{output_name}.txt', 'w') as f:
            for token in game_tokens:
                f.write(f"{token}\n")
