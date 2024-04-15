import torch
from torch import nn, Tensor

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        pass

    def forward(self, x: Tensor) -> Tensor:
        # ToDo: implement
        return self.embedding(x)


def train(data, model, loss_fn, optimizer):
    model.train()
    for batch, (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


def evaluate(data, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss
    return total_loss


def get_batch() -> Tensor:
    """
    Returns batches for the training process
    """
    pass


if __name__ == '__main__':
    from io_utils import get_vocabulary_idx, load_from_file, map_story_to_tensor
    from torchtext.data.utils import get_tokenizer

    stories = load_from_file("data/100stories.txt")
    vocabulary = get_vocabulary_idx(stories)
    tokenizer = get_tokenizer('basic_english')

    model = TransformerModel(len(vocabulary), 100).to(device)  # or torch.load('trained_models/model.pth')
    input = map_story_to_tensor(stories[0], vocabulary, tokenizer)
    pred = model(input)
    print("Embedding:", pred, sep="\n")  # Prints the embedding of the tokens from the first story

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    # torch.save(model, 'model.pth')
