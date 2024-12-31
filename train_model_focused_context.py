import os
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    PreTrainedTokenizerFast
)
import torch
import wandb

wandb.init(project="nba-sim-causal-vocab", entity="mgrimshaw")

# Configuration
TRAIN_FILE = "all_games.txt"
OUTPUT_DIR = "./gpt2-medium-finetuned-focused"  # New output directory
CHECKPOINT_DIR = "./gpt2-medium-finetuned-focused/checkpoint-216000"
BLOCK_SIZE = 48  # Reduced to focus on essential context
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 5e-5
LOGGING_STEPS = 10
SAVE_STEPS = 6000
EVAL_STEPS = 6000
SEED = 42

def extract_game_state(text_line):
    """Extract the game state portion from a line of text"""
    parts = text_line.split('<EVENT_END>')
    if len(parts) > 1:
        # Get the next line's state if available
        return parts[1].strip()
    return text_line.strip()

def create_focused_attention_mask(input_ids, tokenizer):
    """Create an attention mask that focuses on game state and recent events"""
    attention_mask = torch.ones_like(input_ids, dtype=torch.float)
    
    # Find event boundaries
    event_end_token = tokenizer.convert_tokens_to_ids('<EVENT_END>')
    event_ends = (input_ids == event_end_token).nonzero(as_tuple=True)[0]
    
    if len(event_ends) >= 2:
        # Identify the game state tokens and the last two events
        last_event_end = event_ends[-1]
        prev_event_end = event_ends[-2]
        
        # Higher attention (1.5) for:
        # - Current game state
        # - Previous event
        # - Current event
        attention_mask[prev_event_end+1:] = 1.5
        
        # Lower attention (0.5) for older events
        if len(event_ends) > 2:
            attention_mask[:event_ends[-3]] = 0.5
    
    return attention_mask

def group_texts(examples):
    """Group texts into chunks focusing on recent events and game state"""
    concatenated = []
    for arr in examples["input_ids"]:
        concatenated.extend(arr)
    
    # Find event boundaries
    event_end_token = fast_tokenizer.convert_tokens_to_ids('<EVENT_END>')
    event_ends = [i for i, token in enumerate(concatenated) if token == event_end_token]
    
    input_ids_chunks = []
    attention_mask_chunks = []
    
    # Create chunks that include the game state and last 2 events
    for i in range(2, len(event_ends)):
        # Get the start of the previous event
        if i > 2:
            start_idx = event_ends[i-3] + 1
        else:
            start_idx = 0
            
        end_idx = event_ends[i] + 1
        
        # Only create chunk if it fits within BLOCK_SIZE
        if end_idx - start_idx <= BLOCK_SIZE:
            chunk = concatenated[start_idx:end_idx]
            
            # Pad if necessary
            if len(chunk) < BLOCK_SIZE:
                chunk = chunk + [fast_tokenizer.pad_token_id] * (BLOCK_SIZE - len(chunk))
            
            input_ids_chunks.append(chunk)
            
            # Create attention mask
            attention_mask = create_focused_attention_mask(
                torch.tensor(chunk),
                fast_tokenizer
            )
            attention_mask_chunks.append(attention_mask.tolist())
    
    return {
        "input_ids": input_ids_chunks,
        "attention_mask": attention_mask_chunks,
        "labels": input_ids_chunks
    }

if not Path(TRAIN_FILE).is_file():
    print(f"Training file {TRAIN_FILE} not found.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_whitespace_tokenizer.json")

# Load and process dataset
dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

def tokenize_function(examples):
    return fast_tokenizer(
        examples["text"],
        truncation=False,
        add_special_tokens=False,
        return_token_type_ids=False
    )

# Process datasets
tokenized_train = split_dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
grouped_train = tokenized_train.map(group_texts, batched=True)
grouped_train.set_format("torch")

tokenized_eval = split_dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
grouped_eval = tokenized_eval.map(group_texts, batched=True)
grouped_eval.set_format("torch")

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(fast_tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    prediction_loss_only=True,
    seed=SEED,
    report_to="wandb",
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=grouped_train,
    eval_dataset=grouped_eval,
)

# Resume from checkpoint if available
resume_checkpoint = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else None
trainer.train(resume_from_checkpoint=resume_checkpoint)
wandb.finish()

trainer.save_model(OUTPUT_DIR)
fast_tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model and tokenizer saved to", OUTPUT_DIR)