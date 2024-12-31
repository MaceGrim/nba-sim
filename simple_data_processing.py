import json
import csv
import glob
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

class GameEventProcessor:
    """Process NBA game events into structured data"""
    
    def __init__(self):
        self.player_names = set()
        self.team_names = set()
        self.action_types = set()
    
    def parse_game_state(self, state_line: str) -> Dict[str, Any]:
        """Parse a game state line into structured data"""
        parts = state_line.split()
        
        # Extract basic game info
        quarter = parts[0]  # e.g., "Q1"
        time = parts[1]     # e.g., "0:02:51"
        score_home = int(parts[2])
        score_away = int(parts[3])
        home_away = parts[4]  # "Home" or "Away"
        
        # Find indices for team rosters
        home_idx = parts.index("OnCourtHome")
        away_idx = parts.index("OnCourtAway")
        
        # Extract team names and rosters
        home_team = parts[home_idx-1]
        away_team = parts[away_idx-1]
        
        # Get player lists
        home_players = []
        away_players = []
        
        i = home_idx + 1
        while i < len(parts) and not parts[i].startswith("OnCourt"):
            home_players.append(parts[i])
            i += 1
            
        i = away_idx + 1
        while i < len(parts) and not parts[i].startswith("OnCourt"):
            away_players.append(parts[i])
            i += 1
        
        # Update our sets
        self.team_names.update([home_team, away_team])
        self.player_names.update(home_players + away_players)
        
        return {
            "quarter": quarter,
            "time": time,
            "score": {
                "home": score_home,
                "away": score_away
            },
            "home_away": home_away,
            "teams": {
                "home": {
                    "name": home_team,
                    "players": home_players
                },
                "away": {
                    "name": away_team,
                    "players": away_players
                }
            }
        }
    
    def parse_event(self, event_line: str) -> Dict[str, Any]:
        """Parse an event line into structured data"""
        parts = event_line.split()
        
        # Basic event info
        team = parts[0]  # Team code (e.g., "BOS", "PHI")
        self.team_names.add(team)
        
        # Handle different event types
        if "SubOut" in event_line:
            event_type = "Substitution"
            player_out = parts[1]
            player_in = parts[3]
            self.player_names.update([player_out, player_in])
            details = {
                "player_out": player_out,
                "player_in": player_in
            }
        elif "Shot" in event_line:
            event_type = "Shot"
            player = parts[1]
            shot_type = parts[3]  # e.g., "3PT"
            result = parts[-2]  # "Made" or "Missed"
            self.player_names.add(player)
            details = {
                "player": player,
                "shot_type": shot_type,
                "result": result,
                "assist": None if "NoAssist" in event_line else parts[parts.index("Assist")-1],
                "block": None if "NoBlock" in event_line else parts[parts.index("Block")-1]
            }
        elif "FreeThrow" in event_line:
            event_type = "FreeThrow"
            player = parts[1]
            result = parts[-1]
            self.player_names.add(player)
            details = {
                "player": player,
                "result": result
            }
        elif "Rebound" in event_line:
            event_type = "Rebound"
            player = parts[1]
            rebound_type = parts[2]  # "OffensiveRebound" or "DefensiveRebound"
            self.player_names.add(player)
            details = {
                "player": player,
                "rebound_type": rebound_type
            }
        else:
            event_type = "Other"
            details = {"raw_text": event_line}
        
        self.action_types.add(event_type)
        
        return {
            "team": team,
            "type": event_type,
            "details": details
        }
    
    def create_tokenizer(self, save_path: str = "game_tokenizer.json"):
        """Create a simple whitespace tokenizer that preserves the existing format"""
        # Initialize with whitespace pre-tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Train on a vocabulary that includes all observed tokens
        vocab = {
            "<UNK>": 0,
            "<PAD>": 1,
            "<EVENT_END>": 2  # Keep this as it's part of the original format
        }
        
        # Add teams, players, and actions to vocabulary
        for token in sorted(self.team_names):
            vocab[token] = len(vocab)
        
        for token in sorted(self.player_names):
            vocab[token] = len(vocab)
        
        for token in sorted(self.action_types):
            vocab[token] = len(vocab)
        
        # Add common game tokens
        common_tokens = [
            "Home", "Away", "OnCourtHome", "OnCourtAway",
            "SubIn", "SubOut", "Shot", "3PT", "2PT",
            "Made", "Missed", "Rebound", "FreeThrow",
            "OffensiveRebound", "DefensiveRebound",
            "Block", "Assist", "NoAssist", "NoBlock",
            "Q1", "Q2", "Q3", "Q4", "OT"
        ]
        for token in common_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Save tokenizer
        tokenizer.model = WordLevel(vocab, unk_token="<UNK>")
        tokenizer.save(save_path)
        return tokenizer
    
    def process_csv_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Process a single CSV game file into structured events with states"""
        structured_data = []
        
        # Helper functions from process_text.py
        def determine_period_label(period_num):
            if period_num == 1: return "Q1"
            elif period_num == 2: return "Q2"
            elif period_num == 3: return "Q3"
            elif period_num == 4: return "Q4"
            else: return f"OT{period_num-4}"
            
        def determine_shot_type(distance):
            if distance is None: return "2PT"
            return "3PT" if distance > 22 else "2PT"
            
        def underscore_name(name):
            if not name: return name
            return re.sub(r"[^\w]", "_", name.strip())
            
        def parse_teams_from_filename(filename):
            base = os.path.basename(filename)
            parts = base.split('-')
            last_part = parts[-1].replace('.csv', '')
            if '@' in last_part:
                away_team, home_team = last_part.split('@', 1)
                return away_team, home_team
            return "AWAY", "HOME"
        
        # Get team names from filename
        away_team, home_team = parse_teams_from_filename(filepath)
        self.team_names.update([away_team, home_team])
        
        # Track players on court
        away_team_players = set()
        home_team_players = set()
        on_court_away = set()
        on_court_home = set()
        
        # Read all rows first to identify players
        all_rows = []
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                # Track all players
                for player in [row['a1'], row['a2'], row['a3'], row['a4'], row['a5']]:
                    if player:
                        player = underscore_name(player.strip())
                        away_team_players.add(player)
                        on_court_away.add(player)
                for player in [row['h1'], row['h2'], row['h3'], row['h4'], row['h5']]:
                    if player:
                        player = underscore_name(player.strip())
                        home_team_players.add(player)
                        on_court_home.add(player)
        
        self.player_names.update(away_team_players)
        self.player_names.update(home_team_players)
        
        # Process each row
        for i, row in enumerate(all_rows):
            # Create game state
            period_label = determine_period_label(int(row['period']))
            state = {
                "quarter": period_label,
                "time": row['remaining_time'],
                "score": {
                    "home": int(row['home_score']),
                    "away": int(row['away_score'])
                },
                "home_away": "Home",  # Default to home perspective
                "teams": {
                    "home": {
                        "name": home_team,
                        "players": sorted(list(on_court_home))
                    },
                    "away": {
                        "name": away_team,
                        "players": sorted(list(on_court_away))
                    }
                }
            }
            
            # Process event
            event_type = row['event_type'].lower() if row['event_type'] else ""
            event_type_long = row["type"].lower() if row["type"] else ""
            team = row['team'].strip() if row['team'] else ""
            player = underscore_name(row['player']) if row['player'] else None
            
            event = None
            
            if event_type == "shot":
                shot_distance = row['shot_distance'] if row['shot_distance'] else None
                shot_type = determine_shot_type(float(shot_distance) if shot_distance else None)
                event = {
                    "team": team,
                    "type": "Shot",
                    "details": {
                        "player": player,
                        "shot_type": shot_type,
                        "result": "Made" if row['result'].lower() == "made" else "Missed",
                        "assist": underscore_name(row['assist']) if row['assist'] else None,
                        "block": underscore_name(row['block']) if row['block'] else None,
                        "location_x": row['original_x'] if row['original_x'] else None,
                        "location_y": row['original_y'] if row['original_y'] else None
                    }
                }
            
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
                
                event = {
                    "team": team,
                    "type": "Substitution",
                    "details": {
                        "player_in": in_player,
                        "player_out": out_player
                    }
                }
            
            elif event_type == "free throw":
                event = {
                    "team": team,
                    "type": "FreeThrow",
                    "details": {
                        "player": player,
                        "result": "Made" if row['result'].lower() == "made" else "Missed",
                        "number": row['num'],
                        "out_of": row['outof']
                    }
                }
            
            elif event_type == "rebound":
                event = {
                    "team": team,
                    "type": "Rebound",
                    "details": {
                        "player": player,
                        "rebound_type": "Offensive" if "offensive" in event_type_long else "Defensive"
                    }
                }
            
            elif event_type == "turnover":
                event = {
                    "team": team,
                    "type": "Turnover",
                    "details": {
                        "player": player
                    }
                }
            
            elif event_type == "foul":
                event = {
                    "team": team,
                    "type": "Foul",
                    "details": {
                        "player": player,
                        "fouled_player": underscore_name(row['opponent']) if row['opponent'] else None,
                        "foul_type": row['reason'] if row['reason'] else None
                    }
                }
            
            # Add event to structured data if we processed it
            if event:
                self.action_types.add(event["type"])
                structured_data.append({
                    "state": state,
                    "event": event
                })
        
        return structured_data

def process_game_files(self, data_dir: str = "./data") -> List[Dict[str, Any]]:
        """Process all game files in the data directory"""
        all_data = []
        for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
            game_data = self.process_csv_file(filepath)
            all_data.extend(game_data)
        return all_data

def create_dataset(game_data: List[Dict[str, Any]]) -> Dataset:
    """Convert structured game data into a HuggingFace dataset"""
    
    # Convert to pandas DataFrame for easier processing
    rows = []
    for entry in game_data:
        state = entry["state"]
        event = entry["event"]
        
        row = {
            "quarter": state["quarter"],
            "time": state["time"],
            "score_home": state["score"]["home"],
            "score_away": state["score"]["away"],
            "home_away": state["home_away"],
            "home_team": state["teams"]["home"]["name"],
            "away_team": state["teams"]["away"]["name"],
            "home_players": state["teams"]["home"]["players"],
            "away_players": state["teams"]["away"]["players"],
            "event_team": event["team"],
            "event_type": event["type"],
            "event_details": json.dumps(event["details"])  # Convert dict to string
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return Dataset.from_pandas(df)

def main():
    # Initialize processor
    processor = GameEventProcessor()
    
    print("Processing game files from ./data/*.csv")
    # Process all game files
    game_data = processor.process_game_files("./data")
    
    if not game_data:
        print("No game data found in ./data/. Please ensure CSV files are present.")
        return
    
    # Create and save tokenizer
    print("Creating tokenizer...")
    tokenizer = processor.create_tokenizer()
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(game_data)
    
    # Save processed data
    print("Saving processed data...")
    with open("processed_games.json", "w") as f:
        json.dump(game_data, f, indent=2)
    
    # Save dataset
    dataset.save_to_disk("nba_games_dataset")
    
    print(f"\nProcessed data saved to processed_games.json")
    print(f"Dataset saved to nba_games_dataset/")
    print(f"Tokenizer saved to game_tokenizer.json")
    print(f"\nDataset statistics:")
    print(f"Number of events: {len(dataset)}")
    print(f"Number of unique players: {len(processor.player_names)}")
    print(f"Number of teams: {len(processor.team_names)}")
    print(f"Event types: {sorted(processor.action_types)}")

if __name__ == "__main__":
    main()