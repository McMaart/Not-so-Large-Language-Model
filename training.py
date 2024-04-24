import random
import torch
from torch import nn, Tensor
from io_utils import get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel
# import torch.nn.functional as F

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 16
max_seq_len = 32


def train(data: list, model, loss_fn, optimizer, epochs: int = 1, flag_list = None):
    model.train()
    total_loss = 0.
    batch_loss = []

    for epoch in range(1, epochs+1):
        if epoch > 1:
            random.shuffle(data)

        for batch, (x, y) in enumerate(data, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                print("Batch:", batch, f"loss: {total_loss / batch:.6}")
                batch_loss.append(f"Batch: {batch} loss: {total_loss / batch:.6}")

                if flag_list is not None and flag_list[0] == False:
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
    max_idx = min(max_seq_len, data.size(0)) - 1
    return data[:max_idx], data[1:max_idx + 1]


def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
    """
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    return data[:-1], data[1:]


def do_training(end: int = 30000, start: int = 0, flag_list = None):
    stories = load_tiny_stories(end, start)
    stories = clean_stories(stories)
    print("Stories have been loaded")

    vocabulary = get_vocabulary_idx(stories, 1536)
    save_vocabulary(vocabulary)
    # vocabulary_rev = {k: v for v, k in vocabulary.items()}
    tokenizer = get_tokenizer('basic_english')

    # model = torch.load('trained_models/model.pth').to(device)
    model = TransformerModel(len(vocabulary)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_data = [get_sequence(stories, i, vocabulary, tokenizer) for i in range(len(stories))]
    t0 = perf_counter()
    avg_loss, batch_loss = train(train_data, model, loss_fn, optimizer, flag_list = flag_list)
    t = perf_counter() - t0
    print(f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
    print(f"Average Loss: {avg_loss:.5}")
    torch.save(model, 'trained_models/model.pth')
    return t, avg_loss, len(train_data), batch_loss


if __name__ == '__main__':
    do_training(50000)
