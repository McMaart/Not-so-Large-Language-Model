import wandb
from time import perf_counter, sleep
from model_1 import TransformerModel, device
from model_2 import RNNModel
from training import train, evaluate
from io_utils import create_vocabulary, load_tiny_stories, clean_stories, save_vocabulary, load_vocabulary, TinyStories
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

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
        'learning_rate': {'max': 1e-3, 'min': 1e-5},
        'batch_size': {'values': [32, 64, 128, 256]}
    }
}


# Function to handle wandb initialization with retries
def init_wandb_with_retries(config, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            return wandb.init(config=config)
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
    stories = clean_stories(stories)
    vocabulary = create_vocabulary(stories, 2048)
    save_vocabulary(vocabulary)
    data = TinyStories(vocabulary, max_seq_len=256)
    return data, vocabulary


# Calculate evaluation metrics
def calculate_metrics(predictions, targets, vocabulary):
    # Convert tensor to numpy arrays for calculation
    pred_np = predictions.detach().cpu().numpy().astype(int)
    target_np = targets.detach().cpu().numpy().astype(int)

    # Reshape predictions and targets to 2D arrays
    pred_np = pred_np.reshape(pred_np.shape[0], -1)
    target_np = target_np.reshape(target_np.shape[0], -1)

    # Ensure both arrays have the same number of columns
    min_length = min(pred_np.shape[1], target_np.shape[1])
    pred_np = pred_np[:, :min_length]
    target_np = target_np[:, :min_length]

    # Cosine similarity
    cosine_sim = cosine_similarity(pred_np, target_np).mean()

    # Convert indices to tokens for ROUGE-N calculation
    pred_tokens = ["<unk>" if idx >= len(vocabulary) or idx < 0 else vocabulary[idx] for idx in pred_np.flatten()]
    target_tokens = ["<unk>" if idx >= len(vocabulary) or idx < 0 else vocabulary[idx] for idx in target_np.flatten()]

    # ROUGE-N
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(" ".join(target_tokens), " ".join(pred_tokens))

    # Top-k accuracy
    k = 5
    top_k_preds = torch.topk(predictions, k, dim=-1).indices
    top_k_accuracy = (top_k_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    return {
        "cosine_similarity": cosine_sim,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "top_k_accuracy": top_k_accuracy
    }


# Define the function to execute for each sweep trial
def train_transformer_sweep(data, vocabulary, config=None):
    run = init_wandb_with_retries(config=config)
    with run:
        config = wandb.config

        run_name = f"E_{config.embed_size}_L_{config.num_layers}_Dff_{config.dim_ff}_Lr_{config.learning_rate:.7f}_D_{config.dropout}_B_{config.batch_size}"
        wandb.run.name = run_name

        # Initialize the model with hyperparameters from the sweep config
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

        # Measure the start time
        start_time = perf_counter()

        # Train the model
        max_num_batches = 100000  # Define max number of batches
        epoch_losses, avg_loss, batch_losses = train(data, model, loss_fn, optimizer, epochs=1,
                                                     max_num_batches=max_num_batches,
                                                     batch_size=config.batch_size)

        # Measure the end time
        end_time = perf_counter()
        training_time = end_time - start_time
        wandb.log({"training_time": training_time})

        # Log the results for each epoch
        #for epoch, epoch_loss in enumerate(epoch_losses):
           # wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

        # Log the average training loss
        wandb.log({"avg_train_loss": avg_loss})

        # Optionally log batch losses for finer granularity
        for batch_idx, batch_loss in enumerate(batch_losses):
            wandb.log({"batch_idx": batch_idx, "batch_loss": batch_loss})

        # Evaluate the model and log evaluation loss
        eval_loss = evaluate(data, model, loss_fn, max_num_batches=100000)
        wandb.log({"eval_loss": eval_loss})

        # Evaluate additional metrics on a sample
        #dataloader = DataLoader(data, batch_size=1, collate_fn=data.get_batch, num_workers=4, shuffle=True,
                                #pin_memory=True)
        #for x, y in dataloader:
           #x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            #pred = model(x)
            #metrics = calculate_metrics(pred, y, vocabulary)
            #wandb.log(metrics)
            #break  # Log metrics for the first sample only for demonstration purposes


if __name__ == "__main__":
    # Check if user is logged in
    if not wandb.login():
        print("Please log in to your Weights and Biases account using `wandb login` command.")
        exit(1)

    # Prepare data once
    data, vocabulary = prepare_data()

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ml_llm_project')

    # Start the sweep agent
    wandb.agent(sweep_id, function=lambda config=None: train_transformer_sweep(data, vocabulary, config))
