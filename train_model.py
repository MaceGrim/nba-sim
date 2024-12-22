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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Whitespace
import wandb

wandb.init(project="nba-sim-causal-vocab", entity="mgrimshaw")

# Configuration
TRAIN_FILE = "all_games.txt"
OUTPUT_DIR = "./gpt2-medium-finetuned-causal-vocab"
CHECKPOINT_DIR = "./gpt2-medium-finetuned-causal-vocab/checkpoint-216000"  # Example checkpoint
BLOCK_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 5e-5
LOGGING_STEPS = 10
SAVE_STEPS = 6000
EVAL_STEPS = 6000
SEED = 42

if not Path(TRAIN_FILE).is_file():
    print(f"Training file {TRAIN_FILE} not found.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer from custom_whitespace_tokenizer.json
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_whitespace_tokenizer.json")

dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def tokenize_function(examples):
    return fast_tokenizer(
        examples["text"], 
        truncation=False, 
        add_special_tokens=False, 
        return_token_type_ids=False
    )

def group_texts(examples):
    concatenated = []
    for arr in examples["input_ids"]:
        concatenated.extend(arr)
    total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    concatenated = concatenated[:total_length]

    input_ids_chunks = [concatenated[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
    attention_mask_chunks = [[1]*BLOCK_SIZE for _ in input_ids_chunks]

    return {
        "input_ids": input_ids_chunks,
        "attention_mask": attention_mask_chunks,
        "labels": input_ids_chunks
    }

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

if "token_type_ids" in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.remove_columns("token_type_ids")

grouped_train_dataset = tokenized_train_dataset.map(group_texts, batched=True)
grouped_train_dataset.set_format("torch")

tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

if "token_type_ids" in tokenized_eval_dataset.column_names:
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns("token_type_ids")

grouped_eval_dataset = tokenized_eval_dataset.map(group_texts, batched=True)
grouped_eval_dataset.set_format("torch")

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(fast_tokenizer))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Consider removing overwrite_output_dir if you want to preserve the old training outputs
    # overwrite_output_dir=True,
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
    train_dataset=grouped_train_dataset,
    eval_dataset=grouped_eval_dataset,
)

# If the checkpoint directory exists, resume training from that checkpoint
resume_checkpoint = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else None

trainer.train(resume_from_checkpoint=resume_checkpoint)
wandb.finish()

trainer.save_model(OUTPUT_DIR)
fast_tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model and tokenizer saved to", OUTPUT_DIR)
