from datasets import load_from_disk
import wandb
from time import perf_counter, sleep
from model_3 import TransformerModel, device
from training import train, evaluate
from io_utils import TinyStories, load_vocabulary
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchtext.data import get_tokenizer

# Function to handle wandb initialization with retries
def init_wandb_with_retries(config, project_name, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            return wandb.init(config=config, project=project_name)
        except Exception as e:
            print(f"WandB initialization failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {delay} seconds... ({retries}/{max_retries})")
                sleep(delay)
            else:
                raise e

# Load dataset and prepare vocabulary once

def prepare_data(vocab_path):

    # Use the tokenizer and vocabulary directly
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    vocabulary = load_vocabulary(vocab_path)

    # Create the TinyStories dataset objects with the loaded datasets
    train_data = TinyStories(vocabulary, tokenizer, max_seq_len=256, split="train")
    val_data = TinyStories(vocabulary, max_seq_len=256, split="validation")

    return train_data, val_data, vocabulary

# Define the function to train the model
def train_function(config, data, vocabulary, validation_data, project_name, num_epochs=1):
    run = init_wandb_with_retries(config=config, project_name=project_name)
    with run:
        config = wandb.config

        run_name = (f"Ep_{num_epochs}_Spa_Em_{config.embed_size}_L_{config.num_layers}_Dff_{config.dim_ff}"
                    f"_Lr_{config.learning_rate:.5f}_D_{config.dropout}_B_{config.batch_size}_{config.pos_enc_type}")
        wandb.run.name = run_name

        # Initialize the model with hyperparameters from the config
        model = TransformerModel(
            vocab_size=len(vocabulary),
            embed_size=config.embed_size,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_ff=config.dim_ff,
            dropout=config.dropout,
            padding_idx=vocabulary["<pad>"],
            pos_enc_type = config.pos_enc_type
        ).to(device)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        # Number of parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({"num_parameters": params})
        print(f"Model ({params} parameters) and vocabulary ({len(vocabulary)} tokens) have been loaded")

        global best_eval_loss

        total_batches = 0  # Initialize total steps counter

        # Train the model for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Measure the start time
            start_time = perf_counter()

            # Train the model
            max_num_batches = 200000  # Define max number of batches
            avg_loss, batch_losses = train(data, model, loss_fn, optimizer, epochs=1,
                                           max_num_batches=max_num_batches,
                                           batch_size=config.batch_size)

            total_batches += len(batch_losses)  # Update total steps

            # Measure the end time
            end_time = perf_counter()
            training_time = end_time - start_time
            wandb.log({"epoch": epoch + 1, "training_time": training_time})

            # Log the average training loss
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

            # Optionally log batch losses for finer granularity
            for batch_idx, batch_loss in enumerate(batch_losses):
                total_batches += 250  # Increment the batch counter for each batch logged
                wandb.log({"# batches": total_batches, "batch_loss": batch_loss, "epoch": epoch + 1})

        # Evaluate the model on the validation set after all epochs
        print("Starting evaluation...")
        eval_loss = evaluate(validation_data, model, loss_fn, max_num_batches=100000)
        wandb.log({"eval_loss": eval_loss})

        # Save the model if it's the best so far
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model, f'trained_models/best_model.pth')
            print(f"New best model saved with eval loss: {eval_loss}")

# Define the function to execute for each sweep trial
def train_transformer_sweep(data, vocabulary, validation_data, project_name, num_epochs=1):
    # Define the sweep configuration
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'name': 'eval_loss', 'goal': 'minimize'},
        'early_terminate': {'type': 'hyperband', 'max_iter': 8},
        'parameters': {
            'embed_size': {'values': [128, 192]},
            'nhead': {'values': [8]},
            'num_layers': {'values': [1, 2]},
            'dim_ff': {'values': [256, 512]},
            'dropout': {'values': [0.1, 0.2]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0008, 'max': 0.006},
            'batch_size': {'values': [64]},
            'pos_enc_type': {'values': ['sinusoidal', 'rope']}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Start the sweep agent
    wandb.agent(sweep_id, function=lambda config=None: train_function(config, data, vocabulary, validation_data, project_name, num_epochs))

# Define the function to execute the single configuration
def train_transformer_single(data, vocabulary, validation_data, project_name, num_epochs=1):
    # Set the configuration manually
    config = {
        'embed_size': 256,
        'nhead': 8,
        'num_layers': 5,
        'dim_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'pos_enc_type': 'rope'  # 'rope' or 'sinusoidal'
    }

    train_function(config, data, vocabulary, validation_data, project_name, num_epochs)

# Define the function to execute multiple sweeps
def train_transformer_multiple_sweeps(data, vocabulary, validation_data, project_name, num_epochs=1):
    sweep_configuration_1M = {
        'method': 'bayes',
        'metric': {'name': 'eval_loss', 'goal': 'minimize'},
        'early_terminate': {'type': 'hyperband', 'max_iter': 14},
        'parameters': {
            'embed_size': {'values': [128, 192]},
            'nhead': {'values': [8]},
            'num_layers': {'values': [1, 2]},
            'dim_ff': {'values': [128, 256, 512]},
            'dropout': {'values': [0.1, 0.2]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0009, 'max': 0.007},
            'batch_size': {'values': [64]},
            'pos_enc_type': {'values': ['sinusoidal', 'rope']}
        }
    }

    sweep_configuration_5M = {
        'method': 'bayes',
        'metric': {'name': 'eval_loss', 'goal': 'minimize'},
        'early_terminate': {'type': 'hyperband', 'max_iter': 12},
        'parameters': {
            'embed_size': {'values': [256, 384]},
            'nhead': {'values': [8]},
            'num_layers': {'values': [3, 4, 5]},
            'dim_ff': {'values': [256, 512, 1024]},
            'dropout': {'values': [0.1, 0.2]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0007, 'max': 0.005},
            'batch_size': {'values': [64]},
            'pos_enc_type': {'values': ['rope']}
        }
    }
    sweep_configuration_10M = {
        'method': 'bayes',
        'metric': {'name': 'eval_loss', 'goal': 'minimize'},
        'early_terminate': {'type': 'hyperband', 'max_iter': 10},
        'parameters': {
            'embed_size': {'values': [384, 512]},
            'nhead': {'values': [8]},
            'num_layers': {'values': [4, 5]},
            'dim_ff': {'values': [512, 1024, 2048]},
            'dropout': {'values': [0.1, 0.2]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0004, 'max': 0.003},
            'batch_size': {'values': [64]},
            'pos_enc_type': {'values': ['rope']}
        }
    }

    sweep_configuration_15M = {
        'method': 'bayes',
        'metric': {'name': 'eval_loss', 'goal': 'minimize'},
        'early_terminate': {'type': 'hyperband', 'max_iter': 8},
        'parameters': {
            'embed_size': {'values': [512, 768]},
            'nhead': {'values': [8]},
            'num_layers': {'values': [5, 6]},
            'dim_ff': {'values': [2048, 3072, 4096]},
            'dropout': {'values': [0.1, 0.2]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0002, 'max': 0.001},
            'batch_size': {'values': [64]},
            'pos_enc_type': {'values': ['rope']}
        }
    }

    sweep_configs = {
        '1M': sweep_configuration_1M,
        '5M': sweep_configuration_5M,
        '10M': sweep_configuration_10M,
        '15M': sweep_configuration_15M
    }

    for sweep_name, config in sweep_configs.items():
        sweep_id = wandb.sweep(sweep=config, project=f'{project_name}_{sweep_name}')
        wandb.agent(sweep_id, function=lambda config=None: train_function(config, data, vocabulary, validation_data, project_name, num_epochs))


if __name__ == "__main__":
    # Check if user is logged in
    if not wandb.login():
        print("Please log in to your Weights and Biases account using `wandb login` command.")
        exit(1)

    # Prepare data once
    vocab_path = "trained_models/vocabulary.pkl"
    train_data, validation_data, vocabulary = prepare_data(vocab_path)

    # Choose the run type
    run_type = 'single'  # Choose from 'single', 'single_sweep', 'multiple_sweep'

    # Project name
    project_name = 'ml_llm_project'

    # Number of epochs to train
    num_epochs = 2  # Set the desired number of epochs

    # Global variable to track the best evaluation loss
    global best_eval_loss
    best_eval_loss = float('inf')

    match run_type:
        case 'single':
            train_transformer_single(train_data, vocabulary, validation_data, project_name, num_epochs)
        case 'single_sweep':
            train_transformer_sweep(train_data, vocabulary, validation_data, project_name, num_epochs)
        case 'multiple_sweep':
            train_transformer_multiple_sweeps(train_data, vocabulary, validation_data, project_name, num_epochs)