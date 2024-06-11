import sys
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from io_utils import (create_vocabulary, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,
                      load_vocabulary, TinyStories)
from model_1 import TransformerModel, device, learning_rate, max_seq_len
from time import perf_counter
import optuna
from model_2 import RNNModel

writer = SummaryWriter()


def train(data: TinyStories, model: nn.Module, loss_fn, optimizer, epochs: int = 1, max_num_batches: int = None,
          flags: list[bool] = None, batch_size: int = 32, is_rnn: bool = False):
    model.train()
    total_loss = 0.
    curr_loss = 0.
    log_interval = 125
    batch_loss = []
    epoch_losses = []
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.8)

    h = None
    if is_rnn:
        h = model.init_hidden(batch_size)

    # just for IDE
    x: Tensor
    batch, epoch = 1, 1

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.
        shuffle = True  # shuffle = False if epoch == 1 else True
        dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.get_batch, num_workers=4, shuffle=shuffle,
                                pin_memory=True)
        if max_num_batches is None:
            max_num_batches = len(dataloader)

        for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if is_rnn:
                pred, h = model(x, h.detach())
            else:
                pred = model(x)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            loss_item = loss.item()
            total_loss += loss_item
            curr_loss += loss_item
            epoch_loss += loss_item
            writer.add_scalar("Loss/batch", loss_item, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch % log_interval == 0:
                print(f"Batch: {batch:5}, avg. loss: {total_loss / (batch * epoch):.5f},"
                      f" curr. loss: {curr_loss / log_interval:.5f}")
                # ToDO: Adjust the GUI implmentation to receive only the loss (instead of a string)
                batch_loss.append(curr_loss / log_interval)
                curr_loss = 0.

        epoch_loss = epoch_loss / (min(max_num_batches, len(dataloader)))
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    writer.flush()
    writer.close()
    return epoch_losses, total_loss / (max_num_batches * (epoch - 1) + batch), batch_loss


@torch.no_grad()
def evaluate(data: TinyStories, model: nn.Module, loss_fn, max_num_batches: int = 1000, is_rnn: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=2, shuffle=True,
                            pin_memory=True)
    if max_num_batches is None:
        max_num_batches = len(dataloader)

    h = None
    if is_rnn:
        h = torch.zeros(model.num_layers, 32, model.hidden_size).to(device)

    for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if is_rnn:
            pred, h = model(x, h.detach())
        else:
            pred = model(x)

        loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item()
    return total_loss / min(max_num_batches, len(dataloader))


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


def do_training(max_num_batches: int | None = 1000, model_name: str = "model", load_model: bool = True,
                load_vocab: bool = True, flags: list[bool] = None, hyper_search: bool = False):
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
            stories = load_tiny_stories(
                900000)  # Number of stories used for creating the vocabulary, not the vocabulary size
            stories = clean_stories(stories)
            vocabulary = create_vocabulary(stories, 2048)
            save_vocabulary(vocabulary)
        if load_model is True:
            try:
                model = torch.load(f'trained_models/{model_name}.pth').to(device)
            except FileNotFoundError as err:
                print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
                sys.exit(1)
        else:
            model = TransformerModel(len(vocabulary), 192, 6, 6, 768,
                                     0.1, padding_idx=vocabulary["<pad>"]).to(device)
            # model = RNNModel(2048).to(device)
        data = TinyStories(vocabulary, max_seq_len=max_seq_len)
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model ({params} parameters) and vocabulary ({len(vocabulary)} tokens) have been loaded")

        t0 = perf_counter()
        try:
            epoch_losses, avg_loss, batch_loss = train(data, model, loss_fn, optimizer, epochs=1,
                                                       max_num_batches=max_num_batches, flags=flags)
        except KeyboardInterrupt:
            print("Cancelling training, loss statistics will not be available")
            avg_loss = None
            batch_loss = []
        t = perf_counter() - t0
        torch.save(model, f'trained_models/{model_name}.pth')
        if avg_loss is not None:
            print(f"Average Loss: {avg_loss:.5}")
        print(f"Time:{t}")

        return t, avg_loss, max_num_batches, batch_loss


def eval_setup(model_name: str = "model", max_num_batches: int = 1000):
    model = torch.load(f'trained_models/{model_name}.pth').to(device)
    vocabulary = load_vocabulary()
    data = TinyStories(vocabulary, max_seq_len=max_seq_len)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    print(evaluate(data, model, loss_fn, max_num_batches))


def objective(trial):
    # Defines hyperparameter search space
    embed_size = trial.suggest_categorical('embed_size', [256, 512, 768])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dim_ff = trial.suggest_int('dim_ff', 512, 4096, step=256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Load data
    stories = load_tiny_stories(524288)
    stories = clean_stories(stories)
    vocabulary = create_vocabulary(stories, 2048)
    save_vocabulary(vocabulary)
    data = TinyStories(vocabulary, max_seq_len=max_seq_len)

    model = TransformerModel(len(vocabulary), embed_size, nhead, num_layers, dim_ff=dim_ff, dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model
    avg_loss, _ = train(data, model, loss_fn, optimizer, epochs=2, max_num_batches=39500, batch_size=batch_size)
    return avg_loss


if __name__ == '__main__':
    tup = do_training(54000, load_model=False, hyper_search=False, load_vocab=True)
    print(tup)
    print("Starting evaluation...")
    eval_setup("model", max_num_batches=2300)
