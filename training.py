import sys
import torch
from datasets import load_from_disk
from torch import nn, Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from io_utils import create_vocabulary, map_story_to_tensor, save_vocabulary, load_vocabulary, TinyStories
from model_1 import TransformerModel, device, learning_rate, max_seq_len
from time import perf_counter
import optuna
from model_2 import RNNModel, LSTMModel, GRUModel
from torch.cuda.amp import GradScaler
from torch.amp import autocast


def train(data: TinyStories, model: nn.Module, loss_fn, optimizer, epochs: int = 1, max_num_batches: int | None = None,
          batch_size: int = 32, scheduler_stepsize: int = 2500, scheduler_gamma: float = 0.9,
          accumulation_steps: int = 1, max_grad_norm: float = None, flags: list[bool] = None,
          accelerate: bool = False) -> list[float]:
    model.train()
    log_interval = 250
    batch_loss = []
    writer = SummaryWriter()
    scheduler = StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)
    # accelerate = accelerate if device == "cuda" else False
    scaler = GradScaler()

    # just for IDE
    x: Tensor

    for epoch in range(1, epochs + 1):
        curr_loss = 0.0
        shuffle = True  # shuffle = False if epoch == 1 else True
        dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.get_batch, num_workers=2, shuffle=shuffle,
                                pin_memory=True)
        max_num_batches = min(max_num_batches, len(dataloader)) if max_num_batches is not None else len(dataloader)

        for batch, (x, y, lengths) in zip(range(1, max_num_batches + 1), dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if accelerate is True:
                with autocast("cuda"):  # Enable mixed precision
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
                # ToDO: Adjust the GUI implementation to receive only the loss (instead of a string containing the loss)
                batch_loss.append(curr_loss / log_interval)
                curr_loss = 0.0

    writer.flush()
    writer.close()
    return batch_loss


@torch.no_grad()
def evaluate(data: TinyStories, model: nn.Module, loss_fn, max_num_batches: int | None = None,
             use_autocast: bool = True) -> float:
    model.eval()
    total_loss = 0.0
    dataloader = DataLoader(data, batch_size=64, collate_fn=data.get_batch, num_workers=2, shuffle=True,
                            pin_memory=True)
    max_num_batches = min(max_num_batches, len(dataloader)) if max_num_batches is not None else len(dataloader)

    for batch, (x, y, lengths) in zip(range(1, max_num_batches + 1), dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(device, enabled=use_autocast):
            pred = model(x, lengths)
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))

        total_loss += loss.item()
    return total_loss / max_num_batches


def get_batch(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Both input and target Tensor have sizes max_seq_len (for self-attention).
    """
    # ToDo: stack multiple input/target tensor for more efficient training using GPU
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    if len(data) < 2:
        print("Unsuitable data found:", idx, data, story_list[idx], file=sys.stderr)
    max_idx = min(max_seq_len, data.size(0) - 1)
    return data[:max_idx], data[1:max_idx + 1]


def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
    """
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    return data[:-1], data[1:]


def do_training(model_name: str = "model", max_num_batches: int | None = None, load_model: bool = True,
                batch_size: int = 32, load_vocab: bool = True, flags: list[bool] = None, hyper_search: bool = False,
                epochs: int = 1, model_type: str = "transformer") -> list[float]:
    if hyper_search is True:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")
    else:
        if load_vocab is True:
            try:
                vocabulary = load_vocabulary()
            except FileNotFoundError as err:
                print(f"Vocabulary (trained_models/vocabulary.pkl) does not exist!\n{err}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Creating vocabulary...")
            dataset = load_from_disk("data/TinyStories")
            train_stories = dataset["train"][:]["text"]
            vocabulary = create_vocabulary(train_stories, 2048)
            save_vocabulary(vocabulary)

        if load_model is True:
            try:
                model = torch.load(f'trained_models/{model_name}.pth', map_location=device).to(device)
            except FileNotFoundError as err:
                print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
                sys.exit(1)
        else:
            if model_type == "transformer":
                model = TransformerModel(len(vocabulary), 192, 8, 6, 768,
                                         padding_idx=vocabulary["<pad>"])
            elif model_type == "lstm":
                model = LSTMModel(len(vocabulary), padding_idx=vocabulary["<pad>"])
            elif model_type == "rnn":
                model = RNNModel(len(vocabulary), padding_idx=vocabulary["<pad>"])
            else:
                model = GRUModel(len(vocabulary), padding_idx=vocabulary["<pad>"])
            model = model.to(device)

        data = TinyStories(vocabulary, get_tokenizer('spacy', language='en_core_web_sm'), max_seq_len=max_seq_len,
                           split="train")
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model ({params} parameters) and vocabulary ({len(vocabulary)} tokens) have been loaded")

        t0 = perf_counter()
        try:
            batch_loss = train(data, model, loss_fn, optimizer, epochs=epochs,
                               max_num_batches=max_num_batches, flags=flags, batch_size=batch_size)
        except KeyboardInterrupt:
            print("Cancelling training, loss statistics will not be available")
            batch_loss = []
        t = perf_counter() - t0
        torch.save(model, f'trained_models/{model_name}.pth')
        print(f"Time:{t}")

        return batch_loss


def eval_setup(model_name: str = "model", max_num_batches: int | None = None):
    model = torch.load(f'trained_models/{model_name}.pth').to(device)
    vocabulary = load_vocabulary()
    data = TinyStories(vocabulary, max_seq_len=max_seq_len, split="validation")

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    print(f"Evaluation loss: {evaluate(data, model, loss_fn, max_num_batches)}")


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
    batch_loss = train(data, model, loss_fn, optimizer, epochs=1, max_num_batches=100000, batch_size=batch_size)
    newest_batch_loss = batch_loss[-8:]
    return sum(newest_batch_loss) / len(newest_batch_loss)


if __name__ == '__main__':
    model_name = "transformer_model"
    loss_list = do_training(model_name=model_name, max_num_batches=None, load_model=False,
                            load_vocab=True, hyper_search=False, model_type="transformer")
    print(f"Loss list: {loss_list}")
    print("Starting evaluation...")
    eval_setup(model_name)
