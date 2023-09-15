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
import os

# Load all tokenized games
all_tokens = []
# game_tokens = []

game_files = glob("./token_files/*.txt")[:10]  # Add more files if you have them
# game_files = ["./token_files/[2022-10-18]-0022200001-PHI@BOS.txt"]
# for game_file in game_files:
#     with open(game_file, 'r') as f:
#         game_tokens = f.read().splitlines()
#         game_tokens = game_tokens.split('|')
#         all_tokens.extend(game_tokens)

# for game_file in tqdm(game_files, desc="Loading tokens"):
#     with open(game_file, 'r') as f:
#         game_tokens = f.read().splitlines()
#         for line in game_tokens:
#             line_tokens = line.split('|')
#             all_tokens.extend(line_tokens)

game_token_lists = []

for game_file in tqdm(game_files, desc="Loading tokens"):
    with open(game_file, 'r') as f:
        game_tokens = f.read().splitlines()
        game_token_list = []
        for line in game_tokens:
            line_tokens = line.split('|')
            game_token_list.extend(line_tokens)
        game_token_lists.append(game_token_list)
        all_tokens.extend(game_token_list)

# Get max game_token_list length
max_game_token_list_length = 3500
# for game_token_list in game_token_lists:
#     if len(game_token_list) > max_game_token_list_length:
#         max_game_token_list_length = len(game_token_list)

print("Max game token list length: ", max_game_token_list_length)
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

PAD_ID = 0  # or some other value that's not already a token ID

# Define the model configuration
config = GPT2Config(
    vocab_size=len(token_to_id),
    n_positions=3500,  # Maximum sequence length
    n_ctx=500,        # Size of the context window
    n_embd=288,        # Embedding size
    n_layer=12,        # Number of transformer layers
    n_head=12,         # Number of multi-head attention heads
    pad_token_id=PAD_ID     # ID for the padding token (can set it to 0 if not using padding)
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
BATCH_SIZE = 1
EPOCHS = 200
GRADIENT_ACCUMULATION_STEPS = 5
MAX_SEQ_LEN = 3500

# Create PyTorch DataLoader
# input_data = torch.tensor(train_data[:-1], dtype=torch.long)  # All tokens except the last
# target_data = torch.tensor(train_data[1:], dtype=torch.long)  # All tokens except the first

training_length = int(len(game_token_lists)*0.8)

# Create the training dataset
def pad_sequence(sequence, max_length, pad_id=PAD_ID):
    while len(sequence) < max_length:
        sequence.append(pad_id)
    return sequence


input_data, target_data = [], []
for game_token_list in game_token_lists[:training_length]:
    game_ids = [token_to_id[token] for token in game_token_list]

    if len(game_ids) <= MAX_SEQ_LEN:
        # If the sequence is shorter than or equal to MAX_SEQ_LEN,
        # just use the entire sequence and pad it
        input_seq = pad_sequence(game_ids[:-1], MAX_SEQ_LEN)
        target_seq = pad_sequence(game_ids[1:], MAX_SEQ_LEN)

        input_data.append(input_seq)
        target_data.append(target_seq)
    else:
        # If the sequence is longer than MAX_SEQ_LEN, chunk it
        for i in range(0, len(game_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            input_data.append(game_ids[i:i + MAX_SEQ_LEN])
            target_data.append(game_ids[i + 1:i + MAX_SEQ_LEN + 1])


# Print First Tokens
print("First 10 tokens: ", [id_to_token[thing] for thing in input_data[0][:10]])

# Create the validation dataset
input_data_val, target_data_val = [], []
for game_token_list in game_token_lists[training_length:]:
    game_ids = [token_to_id[token] for token in game_token_list]

    if len(game_ids) <= MAX_SEQ_LEN:
        # If the sequence is shorter than or equal to MAX_SEQ_LEN,
        # just use the entire sequence and pad it
        input_seq = pad_sequence(game_ids[:-1], MAX_SEQ_LEN)
        target_seq = pad_sequence(game_ids[1:], MAX_SEQ_LEN)

        input_data_val.append(input_seq)
        target_data_val.append(target_seq)
    else:
        # If the sequence is longer than MAX_SEQ_LEN, chunk it
        for i in range(0, len(game_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            input_data_val.append(game_ids[i:i + MAX_SEQ_LEN])
            target_data_val.append(game_ids[i + 1:i + MAX_SEQ_LEN + 1])

# assert False

input_tensor = torch.tensor(input_data, dtype=torch.long)
target_tensor = torch.tensor(target_data, dtype=torch.long)
dataset = TensorDataset(input_tensor, target_tensor)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation Loader
input_tensor_val = torch.tensor(input_data_val, dtype=torch.long)
target_tensor_val = torch.tensor(target_data_val, dtype=torch.long)
dataset_val = TensorDataset(input_tensor_val, target_tensor_val)

loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

training_losses = []
validation_losses = []

# Training loop
model.train()
train_start_time = str(time.time()).split('.')[0]
os.mkdir(f"./model_checkpoints_{train_start_time}")
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    epoch_start = time.time()
    total_loss = 0.0

    loader_pbar = tqdm(loader, desc="Batches",
                       leave=False)  # leave=False ensures this progress bar gets replaced each epoch
    for batch_idx, (input_batch, target_batch) in enumerate(loader_pbar):

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

        loader_pbar.set_postfix({'Loss': loss.item()})

        # print(scheduler.get_last_lr())

    # Save model checkpoint
    torch.save(model.state_dict(), f"./model_checkpoints_{train_start_time}/model_epoch_{epoch + 1}.pt")

    # Get validation loss
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        # Get validation loss using loader
        loader_val_pbar = tqdm(loader_val, desc="Validation Batches", leave=False)
        for batch_idx, (input_batch, target_batch) in enumerate(loader_val_pbar):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            outputs = model(input_batch, labels=target_batch)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            total_val_loss += loss.item()
            loader_val_pbar.set_postfix({'Loss': loss.item()})

    avg_val_loss = total_val_loss / len(input_data_val)
    validation_losses.append(avg_val_loss)
    model.train()

    # Print epoch results
    avg_loss = total_loss / len(loader)
    training_losses.append(avg_loss)

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f},"
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
