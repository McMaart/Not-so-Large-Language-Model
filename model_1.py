import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 16
max_seq_len = 64


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding: Tensor = self.embedding(x)
        embedding = self.pos_encoding(embedding)
        return self.linear(embedding)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:
        self.eval()
        x = start_token
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1)
            pred = torch.multinomial(probs, 1)[0]
            token_list.append(pred)
            x = pred
        return token_list


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.07):
        super().__init__()
        self.pos_encoding = torch.empty(max_seq_len, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        inv_denominator = 10000**(-1/embed_size*torch.arange(0, embed_size, 2, dtype=torch.float))
        pe_term = pos * inv_denominator
        self.pos_encoding[:, 0::2] = torch.sin(pe_term)
        self.pos_encoding[:, 1::2] = torch.cos(pe_term)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])


if __name__ == '__main__':
    from io_utils import prompt_model, load_vocabulary

    story = prompt_model("model", "there", 40)
    print("\n", story)

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load('trained_models/model.pth')
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    input_tensor = torch.tensor(vocab["once"], dtype=torch.int64).unsqueeze(0)
    tl = model.generate_tokens(input_tensor, 40)
    for val in tl:
        print(vocab_rev[val.item()], end=" ")
