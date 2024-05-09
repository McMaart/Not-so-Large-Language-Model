import random
import sys
import torch
from torch import nn, Tensor
from io_utils import (get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,
                      load_vocabulary)
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel, device, learning_rate, max_seq_len


def train(data: list, model, loss_fn, optimizer, epochs: int = 1, flags: list = None):
    model.train()
    total_loss = 0.
    curr_loss = 0.
    batch_loss = []

    for epoch in range(1, epochs + 1):
        if epoch > 1:
            random.shuffle(data)

        for batch, (x, y) in enumerate(data, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
            curr_loss += loss.item()
            loss.backward()
            if torch.isnan(loss).any():
                print('nan loss at iteration', batch)
                print("Gradient:", model.linear.weight.grad.mean())
                print(f"Prediction {pred}\nTarget: {y}")
            optimizer.step()

            if batch % 500 == 0:
                print(f"Batch: {batch:5}, avg. loss: {total_loss / batch:.5f}, current loss: {curr_loss / 500:.5f}")
                batch_loss.append(f"Batch: {batch} loss: {total_loss / batch:.6}")
                curr_loss = 0.

                if flags is not None and flags[0] is False:
                    return total_loss / len(data), batch_loss

    return total_loss / len(data), batch_loss


def evaluate(data, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(data)


def get_batch(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Both input and target Tensor have sizes max_seq_len (for self-attention).
    """
    # ToDo: stack multiple input/target tensor for more efficient training using GPU
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    if len(data) < 2:
        print("Unsuitable data found:", idx, data, story_list[idx], file=sys.stderr)
    max_idx = min(max_seq_len, data.size(0)-1)
    return data[:max_idx], data[1:max_idx + 1]


def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
    """
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    return data[:-1], data[1:]


def do_training(end: int = 30000, start: int = 0, model_name: str = "model", load_model: bool = True,
                flags: list[bool] = None):
    stories = load_tiny_stories(end, start)
    stories = clean_stories(stories)
    print("Stories have been loaded")

    if load_model is True:
        try:
            vocabulary = load_vocabulary()
            model = torch.load(f'trained_models/{model_name}.pth').to(device)
        except FileNotFoundError as err:
            print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
            sys.exit(1)
    else:
        vocabulary = get_vocabulary_idx(stories, 2048)
        save_vocabulary(vocabulary)
        model = TransformerModel(len(vocabulary)).to(device)

    tokenizer = get_tokenizer('basic_english')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    train_data = [get_batch(stories, i, vocabulary, tokenizer) for i in range(len(stories))]
    t0 = perf_counter()
    avg_loss, batch_loss = train(train_data, model, loss_fn, optimizer, flags=flags)
    t = perf_counter() - t0
    print(f"\nTotal training time: {t:.5}s ({t / len(train_data):.4}s per batch)")
    print(f"Average Loss: {avg_loss:.5}")
    torch.save(model, f'trained_models/{model_name}.pth')

    return t, avg_loss, len(train_data), batch_loss


def eval_setup(model_name: str = "model"):
    # make sure that the model didn't use these stories for training
    stories = load_tiny_stories(1400000, 1200000)
    stories = clean_stories(stories)
    model = torch.load(f'trained_models/{model_name}.pth').to(device)
    vocabulary = load_vocabulary()
    tokenizer = get_tokenizer('basic_english')
    data = [get_batch(stories, i, vocabulary, tokenizer) for i in range(len(stories))]

    loss_fn = nn.CrossEntropyLoss()
    print(evaluate(data, model, loss_fn))


if __name__ == '__main__':
    do_training(end=100000, load_model=False)
