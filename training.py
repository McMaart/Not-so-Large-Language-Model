import random
import sys
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from io_utils import (get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories, save_vocabulary,               load_vocabulary, TinyStories)
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from time import perf_counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from model_1 import TransformerModel, device, learning_rate, max_seq_len, batch_size, embed_size, number_heads, number_layers, dim_ff, dropout

from model_2 import RNNModel, device, learning_rate_rnn, max_seq_len_rnn, batch_size_rnn, embed_size_rnn, hidden_size_rnn, num_layers_rnn, dropout_rnn, dropout_rnn_2
# import torch.nn.functional as F
import optuna
import wandb

epochs = 1
epochs_rnn = 1

def train(data, model, loss_fn, optimizer, epochs: int = 1, max_num_batches: int = None, flags: list = None, batch_size: int = 32, is_rnn: bool = False):
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    model.train()
    total_loss = 0.
    curr_loss = 0.
    log_interval = 100
    batch_loss = []

    for epoch in range(1, epochs + 1):
        shuffle = True  # shuffle = False if epoch == 1 else True
        dataloader = DataLoader(data, batch_size=batch_size, collate_fn=data.get_batch, num_workers=2, shuffle=shuffle, pin_memory=True)
        if max_num_batches is None:
            max_num_batches = len(dataloader)

        h = None
        if is_rnn:
            h = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

        for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
            sys.stdout.write(f"\rBatch {batch + 1}/{min(max_num_batches, len(dataloader))}")
            sys.stdout.flush()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if is_rnn:
                pred, h = model(x, h.detach())
            else:
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
                # Log the current loss to W&B
                wandb.log({'training_loss': curr_loss / log_interval})
                batch_loss.append(curr_loss / log_interval)
                curr_loss = 0.

    return total_loss / (max_num_batches * (epoch - 1) + batch), batch_loss

@torch.no_grad()
def evaluate(data, model, loss_fn, max_num_batches: int = 1000):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=2, shuffle=True,
                            pin_memory=True)
    if max_num_batches is None:
        max_num_batches = len(dataloader)

    for batch, (x, y) in zip(range(1, min(max_num_batches, len(dataloader)) + 1), dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item()

        # Collect all predictions and labels for metrics
        all_preds.append(pred.argmax(dim=-1).view(-1).cpu().numpy())
        all_labels.append(y.view(-1).cpu().numpy())

    avg_loss = total_loss / min(max_num_batches, len(dataloader))
    wandb.log({'evaluation_loss': avg_loss})

    # Calculate additional metrics (precision, recall, f1, accuracy)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    wandb.log({'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy})

    return avg_loss

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
                flags: list[bool] = None, hyper_search: bool = False):
    if hyper_search is True:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")
    else:
        if load_model is True:
            try:
                vocabulary = load_vocabulary()
                model = torch.load(f'trained_models/{model_name}.pth').to(device)
            except FileNotFoundError as err:
                print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Creating vocabulary...")
            stories = load_tiny_stories(
                524288)  # Number of stories used for creating the vocabulary, not the vocabulary size
            stories = clean_stories(stories)
            vocabulary = get_vocabulary_idx(stories, 2048)
            save_vocabulary(vocabulary)
            model = TransformerModel(len(vocabulary), 256, number_heads, 4, 1024, 0.12, padding_idx=vocabulary["<pad>"]).to(device)

        data = TinyStories(vocabulary, max_seq_len=max_seq_len)
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
        print("Model and stories have been loaded")

        t0 = perf_counter()
        try:
            avg_loss, batch_loss = train(data, model, loss_fn, optimizer, epochs=epochs, max_num_batches=max_num_batches, flags=flags)
        except KeyboardInterrupt:
            print("Cancelling training, loss statistics will not be available")
            avg_loss = None
            batch_loss = []
        t = perf_counter() - t0
        wandb.unwatch(model)
        torch.save(model, f'trained_models/{model_name}.pth')
        # Optionally, re-watch the model after saving if needed
        # wandb.watch(model, loss_fn, log="all", log_freq=10)

        if avg_loss is not None:
            print(f"Average Loss: {avg_loss:.5}")
        print(f"Time:{t}")

        return t, avg_loss, max_num_batches, batch_loss
    
def do_training_rnn(max_num_batches: int | None = 1000, model_name: str = "rnn_model", load_model: bool = True, hyper_search: bool = False):
    if hyper_search is True:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_rnn, n_trials=15)
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best hyperparameters: {study.best_trial.params}")
    else:
        if load_model is True:
            try:
                vocabulary = load_vocabulary()
                model = torch.load(f'trained_models/{model_name}.pth').to(device)
            except FileNotFoundError as err:
                print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Creating vocabulary...")
            stories = load_tiny_stories(
                524288)  # Number of stories used for creating the vocabulary, not the vocabulary size
            stories = clean_stories(stories)
            vocabulary = get_vocabulary_idx(stories, 2048)
            save_vocabulary(vocabulary)
            model = RNNModel(len(vocabulary), embed_size=embed_size_rnn, num_layers=num_layers_rnn, dropout_2=dropout_rnn_2, dropout_rnn=dropout_rnn).to(device)

        data = TinyStories(vocabulary, max_seq_len=max_seq_len_rnn)
        loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary["<pad>"])
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate_rnn)
        print("Model and stories have been loaded")

        t0 = perf_counter()
        try:
            avg_loss, batch_loss = train(data, model, loss_fn, optimizer, epochs=epochs, max_num_batches=max_num_batches, flags=None, is_rnn=True)
        except KeyboardInterrupt:
            print("Cancelling training, loss statistics will not be available")
            avg_loss = None
            batch_loss = []
        t = perf_counter() - t0
        wandb.unwatch(model)
        torch.save(model, f'trained_models/{model_name}.pth')
        # Optionally, re-watch the model after saving if needed
        # wandb.watch(model, loss_fn, log="all", log_freq=10)

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

def objective_rnn(trial):
    embed_size_rnn = trial.suggest_categorical("embed_size", 64, 512)

if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1 and argv[1] == 'rnn':
        run_name = f"RNN_{embed_size_rnn}_{num_layers_rnn}_{hidden_size_rnn}_run_{epochs_rnn}_{batch_size_rnn}_{learning_rate_rnn}_{perf_counter()}"
        wandb.login()
        with wandb.init(project='ml_llm_project', name=run_name, config={
            "learning_rate": learning_rate_rnn,
            "epochs_rnn": epochs_rnn,
            "batch_size": batch_size_rnn,
            "embed_size": embed_size_rnn,
            "hidden_size": hidden_size_rnn,
            "num_layers": num_layers_rnn,
            "dropout": dropout_rnn}):
            print(f"Initialized W&B run with name: {run_name}")

            config = wandb.config

            do_training_rnn(7500, model_name=run_name, load_model=False, hyper_search=False)
    else:
        wandb.login()
        run_name = f"run_{epochs}_{batch_size}_{learning_rate}_{perf_counter()}_{number_heads}"
        with wandb.init(project='ml_llm_project', name=run_name, config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "number_heads": number_heads,
            "number_layers": number_layers,
            "dim_ff": dim_ff,
            "dropout": dropout}):
            print(f"Initialized W&B run with name: {run_name}")

            config = wandb.config

            do_training(2000, load_model=False, hyper_search=True)
# Alt für single Batch
# def train(data: list, model, loss_fn, optimizer, epochs: int = 1, flags: list = None):
#     model.train()
#     total_loss = 0.
#     curr_loss = 0.
#     batch_loss = []

#     for epoch in range(1, epochs + 1):
#         if epoch > 1:
#             random.shuffle(data)

#         for batch, (x, y) in enumerate(data, 1):
#             x, y = x.to(device), y.to(device)
#             pred = model(x)

#             optimizer.zero_grad()
#             loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
#             total_loss += loss.item()
#             curr_loss += loss.item()
#             loss.backward()
#             if torch.isnan(loss).any():
#                 print('nan loss at iteration', batch)
#                 print("Gradient:", model.linear.weight.grad.mean())
#                 print(f"Prediction {pred}\nTarget: {y}")
#             optimizer.step()

#             if batch % 500 == 0:
#                 print(f"Batch: {batch:5}, avg. loss: {total_loss / batch:.5f}, current loss: {curr_loss / 500:.5f}")
#                 batch_loss.append(f"Batch: {batch} loss: {total_loss / batch:.6}")
#                 curr_loss = 0.

#                 if flags is not None and flags[0] is False:
#                     return total_loss / len(data), batch_loss

#     return total_loss / len(data), batch_loss

# def train_on_batches(story_list, vocab, tokenizer, model, loss_fn, optimizer, batch_size, device, epochs: int = 1, flags: list = None, is_rnn: bool = False):
#     model.train()  # Set the model to training mode
#     pad_token_id = vocab['<pad>']  # Adjust as per your vocab

#     num_samples = len(story_list)
#     total_batches = num_samples // batch_size
#     batch_loss_list = []

#     for epoch in range(1, epochs + 1):
#         print(f"Starting epoch {epoch}/{epochs}")

#         # Shuffle indices at the beginning of each epoch
#         indices = torch.randperm(num_samples)

#         total_loss = 0.0  # Reset total loss for each epoch

#         # Initialize h if the model is RNN
#         h = None
#         if is_rnn:
#             h = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

#         for batch_index in range(total_batches):
#             sys.stdout.write(f"\rBatch {batch_index + 1}/{total_batches}")
#             sys.stdout.flush()
#             start_index = batch_index * batch_size
#             end_index = start_index + batch_size
#             batch_indices = indices[start_index:end_index]
#             batch_stories = [story_list[i] for i in batch_indices]  # Extract stories for this batch

#             x, y = get_batch(batch_stories, batch_size, vocab, tokenizer)  # Get a batch

#             # Move tensors to the appropriate device
#             x, y = x.to(device), y.to(device)

#             optimizer.zero_grad()  # Clear gradients before each backward pass

#             # Compute predictions
#             if is_rnn:
#                 pred, h = model(x, h.detach())  # Detach h before passing it to the model
#             else:
#                 pred = model(x)

#             mask = (y != pad_token_id) # mask pad token for loss function
#             loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Calculate loss without reduction
#             loss = (loss * mask.view(-1).float()).mean()  # Apply mask and calculate mean loss with mean
#             #loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))  # Compute loss
#             loss.backward()  # Backpropagate the gradients
#             optimizer.step()  # Update model parameters

#             total_loss += loss.item()

#             if (batch_index + 1) % 1000 == 0:
#                 avg_loss = total_loss / (batch_index + 1)
#                 print(f"Batch {batch_index + 1}: Avg. Loss = {avg_loss:.5f}")

#             # Record batch loss for further evaluation or logging
#             batch_loss_list.append(loss.item())

#             if flags is not None and not flags[0]:
#                 break  #noch nicht gecheckt..

#     # Calculate average loss over all batches
#     avg_total_loss = total_loss / total_batches

#     return avg_total_loss, batch_loss_list


# def evaluate(data, model, loss_fn):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for batch, (x, y) in enumerate(data):
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
#             total_loss += loss.item()
#     return total_loss / len(data)


# def get_batch(story_list: list[str], batch_size, vocab, tokenizer) -> tuple[Tensor, Tensor]:
#     """
#     Returns a Tensor of batchsize (input, target) for training.
#     Both input and target Tensor have sizes max_seq_len (for self-attention).
#     """
#     # ohne batches
#     # data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
#     # if len(data) < 2:
#     # print("Unsuitable data found:", idx, data, story_list[idx], file=sys.stderr)
#     # max_idx = min(max_seq_len, data.size(0)) - 1
#     # return data[:max_idx], data[1:max_idx + 1]
#     # ToDo: stack multiple input/target tensor for more efficient training using GPU
#     batch_x = []
#     batch_y = []

#     for i in range(batch_size):
#         data = map_story_to_tensor(story_list[i], vocab, tokenizer)
#         max_idx = min(max_seq_len, data.size(0)) # - 1
#         x = data[:max_idx]
#         y = data[1:max_idx + 1]
#         batch_x.append(x)
#         batch_y.append(y)

#     # Pad sequences and stack them into a single tensor for the batch
#     x_tensor = pad_sequence(batch_x, batch_first=True, padding_value=vocab['<pad>'])
#     y_tensor = pad_sequence(batch_y, batch_first=True, padding_value=vocab['<pad>'])

#     return x_tensor, y_tensor


# def get_sequence(story_list: list[str], idx: int, vocab, tokenizer) -> tuple[Tensor, Tensor]:
#     """
#     Returns a single batch (input, target) for training.
#     Input and target Tensor are independent of max_seq_len (the size is equal to number of tokens - 1)
#     """
#     data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
#     return data[:-1], data[1:]


# def do_training(end: int = 40000, start: int = 0, load_model: bool = True, flags: list = None):
#     stories = load_tiny_stories(end, start)
#     stories = clean_stories(stories)
#     print("Stories have been loaded")

#     if load_model is True:
#         try:
#             vocabulary = load_vocabulary()
#             model = torch.load('trained_models/model.pth').to(device)
#         except FileNotFoundError as err:
#             print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
#             sys.exit(1)
#     else:
#         vocabulary = get_vocabulary_idx(stories, 1536)
#         save_vocabulary(vocabulary)
#         model = TransformerModel(len(vocabulary)).to(device)

#     tokenizer = get_tokenizer('basic_english')
#     loss_fn = nn.CrossEntropyLoss(reduction='none')  # Initialize loss function with 'none' reduction
#     #loss_fn = nn.CrossEntropyLoss()
#     #optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#     optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

#     #Mit Batches
#     #batch_size = 64
#     #print(type(batch))
#     #print(batch[0].size())
#     #print(batch[1].size())
#     #print(batch[0])
#     #print(batch[1])

#     t0 = perf_counter()
#     avg_loss, batch_loss = train_on_batches(stories, vocabulary, tokenizer, model, loss_fn, optimizer, batch_size,
#                                             epochs=1, device=device)
#     t = perf_counter() - t0
#     print(f"\nTraining time: {t:.5}s")
#     print(f"Average Loss: {avg_loss:.5}")
#     print(f"Batch Loss: {batch_loss}")

#     #Alt
#     #train_data = [get_batch(stories, batch_size=2, vocab=vocabulary, tokenizer=tokenizer)for i in range(len(stories))]
#     #train_data = [get_batch(stories, i,  vocabulary, tokenizer) for i in range(len(stories))]
#     #t0 = perf_counter()
#     #avg_loss, batch_loss = train(train_data, model, loss_fn, optimizer, flags=flags)
#     #t = perf_counter() - t0
#     #print(f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
#     #print(f"Average Loss: {avg_loss:.5}")

#     #torch.save(model, 'trained_models/model.pth')
#     torch.save(model, 'trained_models/model2.pth')

#     #return t, avg_loss, len(train_data), batch_loss

# def do_training_rnn(end: int = 1000, start: int = 0, load_model: bool = True, flags: list = None):
#     stories = load_tiny_stories(end, start)
#     stories = clean_stories(stories)
#     print("Stories have been loaded")

#     if load_model is True:
#         try:
#             vocabulary = load_vocabulary()
#             model = torch.load('trained_models/rnn_model.pth').to(device)
#         except FileNotFoundError as err:
#             print(f"Model/vocabulary does not exist!\n{err}", file=sys.stderr)
#             sys.exit(1)
#     else:
#         vocabulary = get_vocabulary_idx(stories, 1536)
#         save_vocabulary(vocabulary)
#         model = RNNModel(len(vocabulary),num_layers=2).to(device)

#     tokenizer = get_tokenizer('basic_english')
#     loss_fn = nn.CrossEntropyLoss(reduction='none')  # Initialize loss function with 'none' reduction
#     optimizer = torch.optim.AdamW(model.parameters(), learning_rate_rnn)

#     t0 = perf_counter()
#     avg_loss, batch_loss = train_on_batches(stories, vocabulary, tokenizer, model, loss_fn, optimizer,device=device, batch_size=batch_size_rnn, epochs=1,  is_rnn=True)
#     t = perf_counter() - t0
#     print(f"\nTraining time: {t:.5}s")
#     print(f"Average Loss: {avg_loss:.5}")
#     print(f"Batch Loss: {batch_loss}")

#     torch.save(model, 'trained_models/rnn_model.pth')

# if __name__ == '__main__':
#     from sys import argv
#     if len(argv) > 1 and argv[1] == 'rnn':
#         do_training_rnn(end = 175000, load_model=False)
#     else:
#         do_training(load_model=False)