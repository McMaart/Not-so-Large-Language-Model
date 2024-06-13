import wandb
from time import perf_counter, sleep
from model_1 import TransformerModel, device
from training import train, evaluate
from io_utils import create_vocabulary, load_tiny_stories, save_vocabulary, TinyStories
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
def prepare_data():
    stories = load_tiny_stories(900000)
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    vocabulary = create_vocabulary(stories, tokenizer, 2048)
    save_vocabulary(vocabulary)
    data = TinyStories(vocabulary, tokenizer, max_seq_len=256)
    return data, vocabulary

# Define the function to train the model
def train_function(config, data, vocabulary, validation_data, project_name, num_epochs=1):
    run = init_wandb_with_retries(config=config, project_name=project_name)
    with run:
        config = wandb.config

        run_name = f"Ep_{num_epochs}_Spa_Em_{config.embed_size}_L_{config.num_layers}_Dff_{config.dim_ff}_Lr_{config.learning_rate:.7f}_D_{config.dropout}_B_{config.batch_size}"
        wandb.run.name = run_name

        # Initialize the model with hyperparameters from the config
        model = TransformerModel(
            vocab_size=len(vocabulary),
            embed_size=config.embed_size,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_ff=config.dim_ff,
            dropout=config.dropout,
            padding_idx=vocabulary["<pad>"]
        ).to(device)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        # Number of parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({"num_parameters": params})
        print(f"Model ({params} parameters) and vocabulary ({len(vocabulary)} tokens) have been loaded")

        global best_eval_loss

        total_steps = 0  # Initialize total steps counter

        # Train the model for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Measure the start time
            start_time = perf_counter()

            # Train the model
            max_num_batches = 100000  # Define max number of batches
            avg_loss, batch_losses = train(data, model, loss_fn, optimizer, epochs=1,
                                           max_num_batches=max_num_batches,
                                           batch_size=config.batch_size)

            total_steps += len(batch_losses)  # Update total steps

            # Measure the end time
            end_time = perf_counter()
            training_time = end_time - start_time
            wandb.log({"epoch": epoch + 1, "training_time": training_time})

            # Log the average training loss
            wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})

            # Optionally log batch losses for finer granularity
            for batch_idx, batch_loss in enumerate(batch_losses):
                wandb.log({"step": total_steps, "batch_loss": batch_loss, "epoch": epoch + 1})

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
        'parameters': {
            'embed_size': {'values': [512, 768, 1024]},
            'nhead': {'values': [4, 8]},
            'num_layers': {'values': [4, 6]},
            'dim_ff': {'values': [512, 1024, 2048]},
            'dropout': {'values': [0.1, 0.2, 0.3]},
            'learning_rate': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-2},
            'batch_size': {'values': [32, 64, 128, 256]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    # Start the sweep agent
    wandb.agent(sweep_id, function=lambda config=None: train_function(config, data, vocabulary, validation_data, project_name, num_epochs))

# Define the function to execute the single configuration
def train_transformer_single(data, vocabulary, validation_data, project_name, num_epochs=1):
    # Set the configuration manually
    config = {
        'embed_size': 768,
        'nhead': 8,
        'num_layers': 6,
        'dim_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 0.00044,
        'batch_size': 64
    }

    train_function(config, data, vocabulary, validation_data, project_name, num_epochs)

if __name__ == "__main__":
    # Check if user is logged in
    if not wandb.login():
        print("Please log in to your Weights and Biases account using `wandb login` command.")
        exit(1)

    # Prepare data once
    data, vocabulary = prepare_data()
    validation_data, _ = prepare_data()  # Load the validation dataset separately

    # Run a sweep or a single configuration
    run_sweep = False  # Set to True if you want to run a hyperparameter sweep

    # Project name
    project_name = 'ml_llm_project'

    # Number of epochs to train
    num_epochs = 3  # Set the desired number of epochs

    # Global variable to track the best evaluation loss
    global best_eval_loss
    best_eval_loss = float('inf')

    if run_sweep:
        # Run the sweep
        train_transformer_sweep(data, vocabulary, validation_data, project_name, num_epochs)
    else:
        # Run single configuration
        train_transformer_single(data, vocabulary, validation_data, project_name, num_epochs)
