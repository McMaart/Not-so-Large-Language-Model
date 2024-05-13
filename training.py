import random
import sys
import torch
from torch import nn, Tensor
from io_utils import (get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,
                      load_vocabulary, TinyStories)
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel, device, learning_rate, max_seq_len
from torch.utils.data import DataLoader


def train(data, model, loss_fn, optimizer, epochs: int = 1, max_num_batches: int = None, flags: list = None):
    model.train()
    total_loss = 0.
    curr_loss = 0.
    log_interval = 500
    batch_loss = []

    # just for IDE
    x: Tensor
    batch, epoch = 1, 1

    for epoch in range(1, epochs + 1):
        shuffle = True # shuffle = False if epoch == 1 else True
        dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=2, shuffle=shuffle,
                                pin_memory=True)
        if max_num_batches is None:
            max_num_batches = len(dataloader)

        for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            if torch.isnan(loss).any():
                print(f"'nan' loss at batch {batch}", file=sys.stderr)
            loss_item = loss.item()
            total_loss += loss_item
            curr_loss += loss_item
            loss.backward()
            optimizer.step()

            if batch % log_interval == 0:
                print(f"Batch: {batch:5}, avg. loss: {total_loss / (batch * epoch):.5f},"
                      f" curr. loss: {curr_loss / log_interval:.5f}")
                # ToDO: append only the batch loss (i.e., adjust gui implementation)
                batch_loss.append(f"Batch: {batch} loss: {total_loss / batch:.6}")
                curr_loss = 0.

    return total_loss / (max_num_batches * (epoch - 1) + batch), batch_loss


@torch.no_grad()
def evaluate(data, model, loss_fn, max_num_batches: int = 1000):
    model.eval()
    total_loss = 0.0
    dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=2, shuffle=True,
                            pin_memory=True)
    if max_num_batches is None:
        max_num_batches = len(dataloader)

    for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                flags: list[bool] = None):

    if load_model is True:
        try:
            vocabulary = load_vocabulary()
            model = torch.load(f'trained_models/{model_name}.pth').to(device)
        except FileNotFoundError as err:
            print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Creating vocabulary...")
        stories = load_tiny_stories(30000)
        stories = clean_stories(stories)
        vocabulary = get_vocabulary_idx(stories, 2048)
        save_vocabulary(vocabulary)
        model = TransformerModel(len(vocabulary)).to(device)

    data = TinyStories(vocabulary, max_seq_len=max_seq_len)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    print("Model and stories have been loaded")

    t0 = perf_counter()
    avg_loss, batch_loss = train(data, model, loss_fn, optimizer, max_num_batches=max_num_batches, flags=flags)
    t = perf_counter() - t0
    print(f"Average Loss: {avg_loss:.5}")
    torch.save(model, f'trained_models/{model_name}.pth')

    return t, avg_loss, max_num_batches, batch_loss


def eval_setup(model_name: str = "model"):
    model = torch.load(f'trained_models/{model_name}.pth').to(device)
    vocabulary = load_vocabulary()
    data = TinyStories(vocabulary)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
    print(evaluate(data, model, loss_fn))


if __name__ == '__main__':
    do_training(4000, load_model=True)
    print("Starting evaluation...")
    eval_setup()
