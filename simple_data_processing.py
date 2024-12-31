import json
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

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
    
    def process_game_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a game file into structured events with states"""
        structured_data = []
        current_state = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if "OnCourt" in line:  # This is a game state line
                    current_state = self.parse_game_state(line)
                elif "<EVENT_END>" in line:  # This is an event line
                    event_text = line.split("<EVENT_END>")[0].strip()
                    event_data = self.parse_event(event_text)
                    
                    # Combine event with current state
                    structured_data.append({
                        "state": current_state,
                        "event": event_data
                    })
        
        return structured_data

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
    
    # Process game file
    game_data = processor.process_game_file("all_games.txt")
    
    # Create and save tokenizer
    tokenizer = processor.create_tokenizer()
    
    # Create dataset
    dataset = create_dataset(game_data)
    
    # Save processed data
    with open("processed_games.json", "w") as f:
        json.dump(game_data, f, indent=2)
    
    # Save dataset
    dataset.save_to_disk("nba_games_dataset")
    
    print(f"Processed data saved to processed_games.json")
    print(f"Dataset saved to nba_games_dataset/")
    print(f"Tokenizer saved to game_tokenizer.json")
    print(f"\nDataset statistics:")
    print(f"Number of events: {len(dataset)}")
    print(f"Number of unique players: {len(processor.player_names)}")
    print(f"Number of teams: {len(processor.team_names)}")
    print(f"Event types: {sorted(processor.action_types)}")

if __name__ == "__main__":
    main()