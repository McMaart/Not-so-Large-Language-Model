import random
import torch
from torch import nn, Tensor
from io_utils import get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel, device, learning_rate, max_seq_len
# import torch.nn.functional as F


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
            optimizer.step()

            if batch % 500 == 0:
                print(f"Batch: {batch:5}, avg. loss: {total_loss / batch:.5f}, current loss: {curr_loss/500:.5f}")
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


def get_batch(story_list: list[str], idx: int,  vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Both input and target Tensor have sizes max_seq_len (for self-attention).
    """
    # ToDo: stack multiple input/target tensor for more efficient training using GPU
    #data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    #max_idx = min(max_seq_len, data.size(0)) - 1
    #indices = torch.randint(low=0, high=max_idx, size=(batch_size, ))
    #X = []
    #Y = []
    #for idx in indices:
        #X.append(data[:max_idx])
       # Y.append(data[1:max_idx + 1])

    #X = torch.stack([torch.nn.functional.pad(t, (0, max_seq_len - t.size(0))) for t in X])
    #Y = torch.stack([torch.nn.functional.pad(t, (0, max_seq_len - t.size(0))) for t in Y])
    # return X, Y

    #for idx in range(batch_size):
     #   data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
      #  max_idx = min(max_seq_len, data.size(0)) - 1
       # x = torch.stack([data[i:i + max_idx] for i in indices])  # input: first maxidx characters starting at i for every i in indices --> stack all 1dimensional tensors as rows
        #y = torch.stack([data[i + 1:i + max_idx + 1] for i in indices])  # target sequence offset by one index
    #return x,y

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


def do_training(end: int = 30000, start: int = 0, flags: list = None):
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

    #batch_size = 512
    #indices = torch.randint(low=0, high=len(stories) - max_seq_len, size=(batch_size,))
    #train_data = get_batch(stories, indices, batch_size, vocabulary, tokenizer)

    train_data = [get_batch(stories, i,  vocabulary, tokenizer) for i in range(len(stories))]
    t0 = perf_counter()
    avg_loss, batch_loss = train(train_data, model, loss_fn, optimizer, flags=flags)
    t = perf_counter() - t0
    print(f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
    print(f"Average Loss: {avg_loss:.5}")
    torch.save(model, 'trained_models/model.pth')

    # eval_stories = load_tiny_stories(120000, 100000)
    # eval_data = [get_batch(eval_stories, i, vocabulary, tokenizer) for i in range(len(eval_stories))]
    # print(evaluate(eval_data, model, loss_fn))
    return t, avg_loss, len(train_data), batch_loss


if __name__ == '__main__':
    do_training()
