"""
Functions for training and evaluating the Transformer/RNN models (ref. Subsection 3.2 in our report).
"""
import sys
from time import perf_counter
from datasets import load_from_disk
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import optuna
from io_utils import create_vocabulary, save_vocabulary, load_vocabulary, TinyStories
from model_1 import TransformerModel, device, max_seq_len
from model_2 import RNNModel, LSTMModel, GRUModel


def train(data: TinyStories, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer, epochs: int = 1,
          batch_size: int = 64, max_num_batches: int | None = None, scheduler_stepsize: int = 2500,
          scheduler_gamma: float = 0.9, accelerate: bool = True, accumulation_steps: int = 1,
          max_grad_norm: float = None, log_interval: int = 250, flags: list[bool] = None) -> list[float]:
    """
    Trains the given model, using the provided dataset.
    :param data: The TinyStories dataset object for training.
    :param model: The instance of the model to be trained.
    :param loss_fn: The loss function used for training (e.g., nn.CrossEntropyLoss).
    :param optimizer: The optimizer used for training the model (e.g., torch.optim.AdamW).
    :param batch_size: The size of each batch.
    :param max_num_batches: The maximum number of batches that will be used. If None, then all available batches
     will be used.
    :param scheduler_stepsize: The number of steps after which the learning rate scheduler updates the learning rate.
    :param scheduler_gamma: The multiplicative factor of learning rate decay.
    :param accelerate: If True, enables mixed precision training.
    :param accumulation_steps: Number of steps to accumulate gradients before performing a backward pass.
    :param max_grad_norm: The maximum norm of gradients. If None, then gradient clipping will be disabled.
    :param log_interval: Every log_interval batches, the average loss of the last log_interval batches is logged.
    The log created by TensorBoard is independent of this parameter (and potentially logs every batch).
    :return: A list with the logged training losses.
    """
    model.train()
    batch_loss = []
    writer = SummaryWriter()
    scheduler = StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)
    scaler = GradScaler(device)

    for epoch in range(1, epochs + 1):
        curr_loss = 0.0
        dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.get_batch, num_workers=1, shuffle=True,
                                pin_memory=True)
        max_num_batches = min(max_num_batches, len(dataloader)) if max_num_batches is not None else len(dataloader)

        for batch, (x, y, lengths) in zip(range(1, max_num_batches + 1), dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if accelerate is True:
                with autocast(device):  # Enable mixed precision
                    pred = model(x, lengths)  # Enable mixed precision
                    loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Enable mixed precision

                scaler.scale(loss).backward()  # Scale the loss - Enable mixed precision
                if batch % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)  # Apply gradients - Enable mixed precision
                    scaler.update()  # Update the scaler - Enable mixed precision
                    scheduler.step()  # Step the scheduler
            else:
                pred = model(x, lengths)
                loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()

            loss_item = loss.item()
            curr_loss += loss_item
            writer.add_scalar("Loss/batch", loss_item, batch)

            if batch % log_interval == 0:
                print(f"Batch: {batch:5}, curr. loss: {curr_loss / log_interval:.5f}")
                batch_loss.append(curr_loss / log_interval)
                curr_loss = 0.0
        max_num_batches -= len(dataloader)

    writer.flush()
    writer.close()
    return batch_loss


@torch.no_grad()
def evaluate(data: TinyStories, model: nn.Module, loss_fn, batch_size: int = 64,
             max_num_batches: int | None = None, use_autocast: bool = True) -> float:
    """
    Evaluates a trained model on the validation or test set.
    :param data: The dataset used for evaluation. Should be an instance of the TinyStories validation or test split.
    :param model: The instance of the model to be evaluated.
    :param loss_fn: The loss function used for evaluation (e.g., nn.CrossEntropyLoss).
    :param batch_size: The size of each batch used for the evaluation.
    :param max_num_batches: The maximum number of batches that will be used. If None,
    then all available batches will be used.
    :param use_autocast: Whether to use Autocasting for the evaluation.
     This can speed up computation time by performing specific operations in lower precision.
    :return: The evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.get_batch, num_workers=2, pin_memory=True)
    max_num_batches = min(max_num_batches, len(dataloader)) if max_num_batches is not None else len(dataloader)

    for batch, (x, y, lengths) in zip(range(1, max_num_batches + 1), dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(device, enabled=use_autocast):
            pred = model(x, lengths)
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item()

    return total_loss / max_num_batches


def training_setup(model_name: str = "model", use_v2: bool = True, load_vocab: bool = True, load_model: bool = True,
                   model_type: str = "transformer", lr: float = 1e-3, epochs: int = 1, batch_size: int = 64,
                   max_num_batches: int | None = None, log_interval: int = 250, accelerate: bool = True,
                   flags: list[bool] = None, hyper_search: bool = False) -> list[float] | None:
    """
    Setup for training a new or existing model.
    :param use_v2: Whether to use the TinyStoriesV2 dataset, which only consists of stories generated by GPT4.
    If False, then the original TinyStories dataset will be used.
    :param model_name: The name of the model to be trained. The model will be saved at
    'trained_models/{model_name}.pth'.
    :param load_vocab: Whether to load a pre-existing vocabulary. If True, the vocabulary will be loaded from
    'trained_models/vocabulary.pkl'. If False, a new vocabulary will be created.
    :param load_model: Whether to load a pre-existing model. If True, the model will be loaded from
    'trained_models/{model_name}.pkl'. If False, a new model of type {model_type} will be created.
    :param model_type: The type of the model that will be newly created. Options are 'transformer', 'rnn', 'gru' and
    'lstm'. The parameter will be ignored if load_model is True.
    :param lr: The initial learning rate.
    :param epochs: The number of training epochs.
    :param batch_size: The size of each training batch.
    :param max_num_batches: The maximum number of batches to train on. If None, train on the full dataset.
    :param log_interval: Every log_interval batches, the average loss of the last log_interval batches is logged.
    Does not apply to the log created by TensorBoard.
    :param accelerate: Enables mixed precision training (gradient scaling + autocasting).
    :return: A list with the logged training losses. Returns None if the training was cancelled or hyper_search is True.
    """
    if hyper_search is True:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")
        return

    dataset_path = "data/TinyStoriesV2" if use_v2 is True else "data/TinyStories"
    if load_vocab is True:
        try:
            vocabulary = load_vocabulary()
        except FileNotFoundError as err:
            print(f"Vocabulary (trained_models/vocabulary.pkl) does not exist!\n{err}", file=sys.stderr)
            sys.exit(1)
    else:
        if load_model is True:
            print(f"Warning: Loading existing model with new vocabulary!", file=sys.stderr)
        print("Creating vocabulary...")
        dataset = load_from_disk(dataset_path)
        train_stories = dataset["train"]["text"]
        vocabulary = create_vocabulary(train_stories, max_words=2048)
        save_vocabulary(vocabulary)
    pad_idx = vocabulary["<pad>"]

    if load_model is True:
        try:
            model = torch.load(f'trained_models/{model_name}.pth', map_location=device, weights_only=False)
        except FileNotFoundError as err:
            print(f"Model 'trained_models/{model_name}.pth' does not exist!\n{err}", file=sys.stderr)
            sys.exit(1)
    else:
        model_type = model_type.lower()
        if model_type == "transformer":
            model = TransformerModel(len(vocabulary), 192, 8, 6, 768,
                                     padding_idx=pad_idx)
        elif model_type == "lstm":
            model = LSTMModel(len(vocabulary), padding_idx=pad_idx)
        elif model_type == "gru":
            model = GRUModel(len(vocabulary), padding_idx=pad_idx)
        else:
            model = RNNModel(len(vocabulary), padding_idx=pad_idx)
        model = model.to(device)

    data = TinyStories(vocabulary, max_seq_len=max_seq_len, split="train", dataset_path=dataset_path)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ({params} parameters) and vocabulary ({len(vocabulary)} tokens) have been loaded")

    batch_loss = None
    t0 = perf_counter()
    try:
        batch_loss = train(data, model, loss_fn, optimizer, epochs=epochs, max_num_batches=max_num_batches,
                           batch_size=batch_size, log_interval=log_interval, flags=flags)
    except KeyboardInterrupt:
        print("Cancelling training...")
    t = perf_counter() - t0
    torch.save(model, f'trained_models/{model_name}.pth')
    print(f"Training time: {t:.2f}s")

    return batch_loss


def evaluation_setup(model_name: str = "model", use_v2: bool = True, batch_size: int = 64,
                     max_num_batches: int | None = None, split: str = "validation") -> float:
    """
    Setup for evaluating a trained model on the validation or test split.
    :param use_v2: Whether to use the TinyStoriesV2 dataset, which only consists of stories generated by GPT4.
    If False, then the original TinyStories dataset will be used.
    :param model_name: The name under which the trained model is stored
    (the corresponding relative file path should be 'trained_models/{model_name}.pth')
    :param batch_size: The batch size for evaluation.
    :param max_num_batches: The maximum number of batches that will be used for the evaluation. If None,
     then all available batches will be used.
    :param split: The split used for evaluation. Options are 'validation' and 'test'.
    :return: The evaluation loss.
    """
    model = torch.load(f'trained_models/{model_name}.pth', map_location=device, weights_only=False).to(device)
    vocabulary = load_vocabulary()
    dataset_path = "data/TinyStoriesV2" if use_v2 is True else "data/TinyStories"
    eval_data = TinyStories(vocabulary, max_seq_len=max_seq_len, split=split, dataset_path=dataset_path)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    eval_loss = evaluate(eval_data, model, loss_fn, batch_size, max_num_batches)
    print(f"Evaluation loss: {eval_loss}")
    return eval_loss


def objective(trial):
    # Defines hyperparameter search space
    embed_size = trial.suggest_categorical('embed_size', [256, 512, 768])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dim_ff = trial.suggest_int('dim_ff', 512, 4096, step=256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    pos_enc_type = trial.suggest_categorical('pos_enc_type', ['sinusoidal', 'rope'])

    # Load data
    vocabulary = load_vocabulary()
    data = TinyStories(vocabulary, max_seq_len=max_seq_len)

    model = TransformerModel(len(vocabulary), embed_size, nhead, num_layers, dim_ff=dim_ff, dropout=dropout,
                             pos_enc_type=pos_enc_type).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model
    _ = train(data, model, loss_fn, optimizer, epochs=1, max_num_batches=100000, batch_size=batch_size)
    evaluate_data = TinyStories(vocabulary, max_seq_len=max_seq_len, split="validation")
    return evaluate(evaluate_data, model, loss_fn)


if __name__ == '__main__':
    model_name = "transformer_8.3M_V2"
    training_setup(model_name=model_name, model_type="transformer", load_model=False, lr=1e-3)
    print("Starting evaluation...")
    evaluation_setup(model_name)
