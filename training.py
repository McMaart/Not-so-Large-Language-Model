import random
import sys
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from io_utils import (get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,
                      load_vocabulary)
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel, device, learning_rate, max_seq_len
# import torch.nn.functional as F

# Alt für single Batch
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

def train_on_batches(story_list, vocab, tokenizer, model, loss_fn, optimizer, batch_size, device, epochs: int = 1, flags: list = None):
    model.train()  # Set the model to training mode

    num_samples = len(story_list)
    total_batches = num_samples // batch_size
    batch_loss_list = []

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}")

        # Shuffle indices at the beginning of each epoch
        indices = torch.randperm(num_samples)

        total_loss = 0.0  # Reset total loss for each epoch

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            batch_indices = indices[start_index:end_index]
            batch_stories = [story_list[i] for i in batch_indices]  # Extract stories for this batch

            x, y = get_batch(batch_stories, batch_size, vocab, tokenizer)  # Get a batch

            # Move tensors to the appropriate device
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # Clear gradients before each backward pass
            pred = model(x)  # Compute predictions
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Compute loss
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update model parameters

            total_loss += loss.item()

            if (batch_index + 1) % 15 == 0:
                avg_loss = total_loss / (batch_index + 1)
                print(f"Batch {batch_index + 1}: Avg. Loss = {avg_loss:.5f}")

            # Record batch loss for further evaluation or logging
            batch_loss_list.append(loss.item())

            if flags is not None and not flags[0]:
                break  #noch nicht gecheckt..

    # Calculate average loss over all batches
    avg_total_loss = total_loss / total_batches

    return avg_total_loss, batch_loss_list


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


def get_batch(story_list: list[str], batch_size, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a Tensor of batchsize (input, target) for training.
    Both input and target Tensor have sizes max_seq_len (for self-attention).
    """
    # ohne batches
    # data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    # if len(data) < 2:
    # print("Unsuitable data found:", idx, data, story_list[idx], file=sys.stderr)
    # max_idx = min(max_seq_len, data.size(0)) - 1
    # return data[:max_idx], data[1:max_idx + 1]
    # ToDo: stack multiple input/target tensor for more efficient training using GPU
    batch_x = []
    batch_y = []

    for i in range(batch_size):
        data = map_story_to_tensor(story_list[i], vocab, tokenizer)
        max_idx = min(max_seq_len, data.size(0)) - 1
        x = data[:max_idx]
        y = data[1:max_idx + 1]
        batch_x.append(x)
        batch_y.append(y)

    # Pad sequences and stack them into a single tensor for the batch
    x_tensor = pad_sequence(batch_x, batch_first=True, padding_value=vocab['<pad>'])
    y_tensor = pad_sequence(batch_y, batch_first=True, padding_value=vocab['<pad>'])

    return x_tensor, y_tensor


def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
    """
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    return data[:-1], data[1:]


def do_training(end: int = 30000, start: int = 0, load_model: bool = True, flags: list = None):
    stories = load_tiny_stories(end, start)
    stories = clean_stories(stories)
    print("Stories have been loaded")

    if load_model is True:
        try:
            vocabulary = load_vocabulary()
            model = torch.load('trained_models/model.pth').to(device)
        except FileNotFoundError as err:
            print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
            sys.exit(1)
    else:
        vocabulary = get_vocabulary_idx(stories, 1536)
        save_vocabulary(vocabulary)
        model = TransformerModel(len(vocabulary)).to(device)

    tokenizer = get_tokenizer('basic_english')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    #Mit Batches
    batch_size = 512
    #print(type(batch))
    #print(batch[0].size())
    #print(batch[1].size())
    #print(batch[0])
    #print(batch[1])

    t0 = perf_counter()
    avg_loss, batch_loss = train_on_batches(stories, vocabulary, tokenizer, model, loss_fn, optimizer, batch_size,
                                            epochs=5, device=device)
    t = perf_counter() - t0
    print(f"\nTraining time: {t:.5}s")
    print(f"Average Loss: {avg_loss:.5}")
    print(f"Batch Loss: {batch_loss}")

    #Alt
    #train_data = [get_batch(stories, batch_size=2, vocab=vocabulary, tokenizer=tokenizer)for i in range(len(stories))]
    #train_data = [get_batch(stories, i,  vocabulary, tokenizer) for i in range(len(stories))]
    #t0 = perf_counter()
    #avg_loss, batch_loss = train(train_data, model, loss_fn, optimizer, flags=flags)
    #t = perf_counter() - t0
    #print(f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
    #print(f"Average Loss: {avg_loss:.5}")

    #torch.save(model, 'trained_models/model.pth')

    #return t, avg_loss, len(train_data), batch_loss


if __name__ == '__main__':
    do_training(load_model=False)
