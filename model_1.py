import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3
# batch_size = 16
max_seq_len = 16


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.embedding(x)
        return self.linear(embedding)


def train(data: list, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.
    batch_list = []
    for batch, (x, y) in enumerate(data, 1):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        optimizer.zero_grad()
        loss = loss_fn(pred.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            print("Batch:",batch, f"loss: {total_loss / batch:.6}")
            batch_list.append(f"Batch: {batch} loss: {total_loss / batch:.6}")

    return total_loss / len(data), batch_list


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
    Returns a batch for the training process
    """
    from io_utils import map_story_to_tensor
    data = map_story_to_tensor(story_list[idx], vocab, tokenizer)
    max_idx = min(max_seq_len, data.size(0)) - 1
    x = data[:max_idx]
    y = data[1:max_idx + 1]
    return x, y

def do_training():
    from io_utils import get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories
    from torchtext.data.utils import get_tokenizer
    from time import perf_counter

    stories = load_tiny_stories(20000)
    stories = clean_stories(stories)
    vocabulary = get_vocabulary_idx(stories)
    vocabulary_rev = {k: v for v, k in vocabulary.items()}
    tokenizer = get_tokenizer('basic_english')

    # model = torch.load('trained_models/model.pth').to(device)
    model = TransformerModel(len(vocabulary)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_data = [get_batch(stories, i, vocabulary, tokenizer) for i in range(19000)]
    t0 = perf_counter()
    avg_loss,l = train(train_data, model, loss_fn, optimizer)
    t = perf_counter() - t0
    print(f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
    print(f"Average Loss: {avg_loss:.5}")
    print(f"Average Loss: {avg_loss:.5f}")
    # torch.save(model, 'trained_models/model.pth')
    return t, avg_loss, len(train_data),l

if __name__ == '__main__':
    do_training()
