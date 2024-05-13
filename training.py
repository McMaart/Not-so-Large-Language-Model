import random
import sys
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from io_utils import (get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,
                      load_vocabulary)
from torchtext.data.utils import get_tokenizer
from time import perf_counter
from model_1 import TransformerModel, device, learning_rate, max_seq_len , batch_size
# import torch.nn.functional as F

def train_on_batches(story_list, vocab, tokenizer, model, loss_fn, optimizer, batch_size, device, epochs: int = 1, flags: list = None):
    model.train()  # Set the model to training mode
    pad_token_id = vocab['<pad>']  # Adjust as per your vocab

    num_samples = len(story_list)
    total_batches = num_samples // batch_size
    has_remaining_batch = num_samples % batch_size != 0  # Check if there's a remaining batch
    batch_loss_list = []

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}")

        # Shuffle indices at the beginning of each epoch
        indices = torch.randperm(num_samples)

        total_loss = 0.0  # Reset total loss for each epoch

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size

            if batch_index == total_batches and has_remaining_batch:  # Handle remaining data points
                batch_indices = indices[start_index:]  # Use remaining indices
            else:
                batch_indices = indices[start_index:end_index]

            #batch_indices = indices[start_index:end_index]
            batch_stories = [story_list[i] for i in batch_indices]  # Extract stories for this batch

            x, y = get_batch(batch_stories, vocab, tokenizer)  # Get a batch

            # Move tensors to the appropriate device
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # Clear gradients before each backward pass
            pred = model(x)  # Compute predictions
            mask = (y != pad_token_id)  # mask pad token for loss function
            loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Calculate loss without reduction
            loss = (loss * mask.view(-1).float()).mean()  # Apply mask and calculate mean loss with mean
            #loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Compute loss
            loss.backward()  # Backpropagate the gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update model parameters

            total_loss += loss.item()

            if (batch_index + 1) % 150 == 0:
                avg_loss = total_loss / (batch_index + 1)
                print(f"Batch {batch_index + 1}: Avg. Loss = {avg_loss:.5f}")

            # Record batch loss for further evaluation or logging
            batch_loss_list.append(loss.item())

            if flags is not None and not flags[0]:
                break  #noch nicht gecheckt..

    # Calculate average loss over all batches
    #avg_total_loss = total_loss / total_batches

    #return avg_total_loss, batch_loss_list
    avg_total_loss = total_loss / (total_batches + (1 if has_remaining_batch else 0))
    return avg_total_loss #max_idx


def evaluate(data, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y.view(-1))
            #loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(data)


def get_batch(batch_stories: list[str], vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a Tensor of batchsize (input, target) for training.
    Both input and target Tensor have sizes max_seq_len (for self-attention).
    """
    # ToDo: stack multiple input/target tensor for more efficient training using GPU
    batch_x = []
    batch_y = []

    for story in batch_stories:
        data = map_story_to_tensor(story, vocab, tokenizer)
        max_idx = min(max_seq_len, data.size(0) - 1)
        x = data[:max_idx]
        y = data[1:max_idx + 1]
        batch_x.append(x)
        batch_y.append(y)

    # Pad sequences and stack them into a single tensor for the batch
    x_tensor = pad_sequence(batch_x, batch_first=True, padding_value=vocab['<pad>'])
    y_tensor = pad_sequence(batch_y, batch_first=True, padding_value=vocab['<pad>'])

    return x_tensor, y_tensor, #max_idx


def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
    """
    Returns a single batch (input, target) for training.
    Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
    """
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    return data[:-1], data[1:]


def do_training(end: int = 200000, start: int = 0, load_model: bool = True, flags: list = None):
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
        vocabulary = get_vocabulary_idx(stories, 3000)
        save_vocabulary(vocabulary)
        model = TransformerModel(len(vocabulary)).to(device)

    #print(f"Story 1: {stories[len(stories)-1]}")
    #n = int(0.9*len(stories))
    #train_data = stories[:n]
    #val_data = stories[n:]
    #print(f"Train_data: {train_data}")
    #print(f"Val_Data: {val_data}")

    tokenizer = get_tokenizer('basic_english')
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Initialize loss function with 'none' reduction
    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    t0 = perf_counter()

    avg_loss = train_on_batches(stories, vocabulary, tokenizer, model, loss_fn, optimizer, batch_size,
                                            epochs=5, device=device)
    t = perf_counter() - t0
    print(f"\nTraining time: {t:.5}s")
    print(f"Average Loss: {avg_loss:.5}")


    #torch.save(model, 'trained_models/model.pth')
    torch.save(model, 'trained_models/model2.pth')

    #return t, avg_loss, len(train_data), batch_loss


if __name__ == '__main__':
    do_training(load_model=False)