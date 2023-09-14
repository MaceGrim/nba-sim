import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import time

# Load all tokenized games
all_tokens = []
# game_tokens = []

game_files = glob("./token_files/*.txt")  # Add more files if you have them
# game_files = ["./token_files/[2022-10-18]-0022200001-PHI@BOS.txt"]
# for game_file in game_files:
#     with open(game_file, 'r') as f:
#         game_tokens = f.read().splitlines()
#         game_tokens = game_tokens.split('|')
#         all_tokens.extend(game_tokens)

for game_file in tqdm(game_files, desc="Loading tokens"):
    with open(game_file, 'r') as f:
        game_tokens = f.read().splitlines()
        for line in game_tokens:
            line_tokens = line.split('|')
            all_tokens.extend(line_tokens)

# for game_file in game_files[:5]:
#     curr_tokens = []
#     with open(game_file, 'r') as f:
#         game_tokens = f.read().splitlines()
#         curr_tokens.extend(game_tokens)
#     game_tokens.append(curr_tokens)

print("Length of all tokens: ", len(set(all_tokens)))
print("Number of Games: ", len(game_files))

# Create a vocabulary and map each token to a unique integer ID
token_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
id_to_token = {idx: token for token, idx in token_to_id.items()}

# Convert all tokens to their respective IDs
all_ids = [token_to_id[token] for token in all_tokens]

# Split data into training and validation sets
train_size = int(0.8 * len(all_ids))
train_data = all_ids[:train_size]
val_data = all_ids[train_size:]

# assert False

# Save the token-to-ID mapping for future reference
with open("token_to_id.txt", 'w') as f:
    for token, idx in token_to_id.items():
        f.write(f"{token}\t{idx}\n")

print("Data preparation completed!")

# Define the model configuration
config = GPT2Config(
    vocab_size=len(token_to_id),
    n_positions=600,  # Maximum sequence length
    n_ctx=500,        # Size of the context window
    n_embd=12*8,        # Embedding size
    n_layer=12,        # Number of transformer layers
    n_head=12,         # Number of multi-head attention heads
    pad_token_id=0     # ID for the padding token (can set it to 0 if not using padding)
)

# Instantiate the model
model = GPT2LMHeadModel(config)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=3e-4)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data))

print("Model defined and ready for training!")

# Hyperparameters
# BATCH_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 240
GRADIENT_ACCUMULATION_STEPS = 1
MAX_SEQ_LEN = 600

# Create PyTorch DataLoader
# input_data = torch.tensor(train_data[:-1], dtype=torch.long)  # All tokens except the last
# target_data = torch.tensor(train_data[1:], dtype=torch.long)  # All tokens except the first

# Create the training dataset
input_data, target_data = [], []
for i in range(0, len(train_data) - MAX_SEQ_LEN, MAX_SEQ_LEN):
    input_data.append(train_data[i:i+MAX_SEQ_LEN])
    target_data.append(train_data[i+1:i+MAX_SEQ_LEN+1])
input_data = torch.tensor(input_data, dtype=torch.long)
target_data = torch.tensor(target_data, dtype=torch.long)

# Create the validation dataset
val_input_data, val_target_data = [], []
for i in range(0, len(val_data) - MAX_SEQ_LEN, MAX_SEQ_LEN):
    val_input_data.append(val_data[i:i+MAX_SEQ_LEN])
    val_target_data.append(val_data[i+1:i+MAX_SEQ_LEN+1])
val_input_data = torch.tensor(val_input_data, dtype=torch.long)
val_target_data = torch.tensor(val_target_data, dtype=torch.long)


dataset = TensorDataset(input_data, target_data)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

training_losses = []
validation_losses = []

# Training loop
model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0.0
    for batch_idx, (input_batch, target_batch) in enumerate(loader):

        # Move data to GPU
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        # Forward pass
        outputs = model(input_batch, labels=target_batch)
        loss = outputs.loss
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            # scheduler.step()
            model.zero_grad()

        # print(scheduler.get_last_lr())

    # Save model checkpoint
    torch.save(model.state_dict(), f"./model_checkpoints/model_epoch_{epoch + 1}.pt")

    # Get validation loss
    model.eval()
    with torch.no_grad():
        val_input_data, val_target_data = val_input_data.to(device), val_target_data.to(device)
        outputs = model(val_input_data, labels=val_target_data)
        val_loss = outputs.loss
        validation_losses.append(val_loss.item())

    model.train()

    # Print epoch results
    avg_loss = total_loss / len(loader)
    training_losses.append(avg_loss)

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f},"
          f" Time: {time.time() - epoch_start:.4f}s")

print("Training complete!")

# Everything after this is for validation and can be ignored for now. I'm cleaning it up into sepearate files.
# _____________________________________________________________________________

plt.plot(training_losses)
plt.plot(validation_losses)
plt.legend(["Training Loss", "Validation Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Hyperparameters for validation
VAL_BATCH_SIZE = 32

# Create PyTorch DataLoader for validation data
# val_input_data = torch.tensor(val_data[:-1], dtype=torch.long)  # All tokens except the last
# val_target_data = torch.tensor(val_data[1:], dtype=torch.long)  # All tokens except the first

# val_dataset = TensorDataset(val_input_data, val_target_data)
# val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

# Switch model to evaluation mode
model.eval()

# Initialize loss
# total_val_loss = 0.0
#
# # No need to compute gradients during validation
# with torch.no_grad():
#     for batch_idx, (input_batch, target_batch) in enumerate(val_loader):
#         # Move data to GPU
#         input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#
#         # Forward pass
#         outputs = model(input_batch, labels=target_batch)
#         loss = outputs.loss
#         total_val_loss += loss.item()
#
# # Compute average validation loss
# avg_val_loss = total_val_loss / len(val_loader)

# Compute perplexity (optional)
# perplexity = torch.exp(torch.tensor(avg_val_loss))
#
# print(f"Validation Loss: {avg_val_loss:.4f}")
# print(f"Perplexity: {perplexity:.4f}")


def generate_sequence(model, starting_tokens, max_length=1000, top_k=1):
    """
    Generate a sequence using the trained model.

    Parameters:
    - model: the trained GPT-2 model.
    - starting_tokens: List of tokens to start generation with.
    - max_length: maximum number of tokens in the generated sequence.
    - top_k: number of top tokens to sample from.

    Returns:
    - A list of generated tokens.
    """

    # Convert starting tokens to their corresponding IDs
    input_ids = torch.tensor([token_to_id[token] for token in starting_tokens], dtype=torch.long).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Start with the provided starting tokens
    generated_sequence = list(starting_tokens)

    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions for the next token
            outputs = model(input_ids)

            predictions = outputs.logits[0, -1, :]  # We only need the predictions of the last token

            # Sample from the top k tokens
            top_k_tokens = torch.topk(predictions, k=top_k).indices

            next_token_id = top_k_tokens[torch.randint(0, top_k, (1,))].item()

            # Convert ID back to token and append to the generated sequence
            next_token = id_to_token[next_token_id]
            generated_sequence.append(next_token)

            # Check for end token
            if next_token == "END_OF_GAME":
                break

            # Add the new token ID to the input for the next iteration
            new_input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_ids = torch.cat((input_ids, new_input_ids), dim=1)

    return generated_sequence



# Generate a sequence starting with a specific token (e.g., "START_OF_GAME")

starting_tokens = [
    "START_OF_GAME",
    "AWAY_START|Joel Embiid",
    "AWAY_START|Tobias Harris",
    "AWAY_START|P.J. Tucker",
    "AWAY_START|Tyrese Maxey",
    "AWAY_START|James Harden",
    "AWAY_BENCH|Montrezl Harrell",
    "AWAY_BENCH|Danuel House Jr.",
    "AWAY_BENCH|Georges Niang",
    "AWAY_BENCH|Matisse Thybulle",
    "AWAY_BENCH|De'Anthony Melton",
    "HOME_START|Al Horford",
    "HOME_START|Jaylen Brown",
    "HOME_START|Marcus Smart",
    "HOME_START|Jayson Tatum",
    "HOME_START|Derrick White",
    "HOME_BENCH|Noah Vonleh",
    "HOME_BENCH|Blake Griffin",
    "HOME_BENCH|Malcolm Brogdon",
    "HOME_BENCH|Sam Hauser",
    "HOME_BENCH|Grant Williams"
]


generated_tokens = generate_sequence(model, starting_tokens)
print(generated_tokens)

with open(f'./sample_gen.txt', 'w') as f:
    for token in generated_tokens:
        f.write(f"{token}\n")

lengths = []
for i in tqdm(range(300)):
    generated_tokens = generate_sequence(model, starting_tokens)
    lengths.append(len(generated_tokens))
sum(lengths) / len(lengths)
plt.hist(lengths, bins=100)
plt.show()


def player_validity_metric(generated_sequence):
    """
    Calculate the number of times a non-active or benched player is mentioned in the generated sequence.

    Parameters:
    - generated_sequence: List of tokens from the generated game sequence.

    Returns:
    - Invalid player mentions count.
    - Benched player mentions count.
    """

    def extract_players_from_sub_token(token):
        """
        Extract the names of the players involved in a substitution token.

        Parameters:
        - token: A substitution token.

        Returns:
        - The names of the players involved in the substitution.
        """

        # Assuming a substitution token structure like "SUB|PLAYER_OUT|PLAYER_IN"
        # This extraction process depends on your token structure
        player_in = token.split("|")[-1]
        player_out = token.split("|")[0]

        return player_in, player_out

    invalid_mentions = 0
    benched_mentions = 0

    # Extracting starting lineup and benched players from context tokens
    # Assuming the structure "STARTING_PLAYER_X" for starting players
    active_players = [token.split("|")[-1] for token in generated_sequence if "_START|" in token]
    benched_players = [token.split("|"[-1]) for token in generated_sequence if "_BENCH|" in token]

    for token in generated_sequence:
        # Assuming player tokens have a specific prefix or structure. Adjust as needed.
        first_piece = token.split("|")[0]
        if first_piece not in ("START_OF_GAME", "END_OF_GAME", "SUB", "nan"):
            if first_piece not in (active_players + benched_players):
                invalid_mentions += 1
            elif first_piece in benched_players:
                benched_mentions += 1
        # Assuming a substitution token structure like "SUB_IN_PLAYER_X_FOR_PLAYER_Y"
        elif "SUB_IN" in token:
            # Extract player names from the token
            # This extraction process depends on your token structure
            player_in, player_out = extract_players_from_sub_token(token)
            active_players.append(player_in)
            benched_players.append(player_out)
            active_players.remove(player_out)
            benched_players.remove(player_in)

    return invalid_mentions, benched_mentions


print(player_validity_metric(generated_tokens))


def aggregate_scores(generated_sequence):
    """
    Aggregate scores for each player and team from the generated sequence.

    Parameters:
    - generated_sequence: List of tokens from the generated game sequence.

    Returns:
    - Dictionary with player scores.
    - Dictionary with team scores.
    """

    player_scores = defaultdict(int)
    team_scores = defaultdict(int)

    for token in generated_sequence:
        # Assuming tokens have a structure like "ACTION|PLAYER|TEAM|POINTS"
        pieces = token.split("|")

        # If it's a scoring action
        if len(pieces) > 2:
            if pieces[-2] == "made":  # Adjust based on your token design
                player = pieces[0]
                team = pieces[-1]

                points = 0
                shot_type = pieces[1]
                if shot_type == "shot":
                    points = 2
                elif shot_type == "free throw":
                    points = 1
                elif shot_type == "3pt Shot":
                    points = 3

                player_scores[player] += points
                team_scores[team] += points

    return player_scores, team_scores

print(aggregate_scores(generated_tokens))
