import torch
from torch import nn, Tensor, TensorType
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
        #self.mha = MultiHeadAttenion(model_dim, num_heads)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding: Tensor = self.embedding(x)
        embedding = self.pos_encoding(embedding).to(device)
        return self.linear(embedding)

    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:
        x = start_token
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1)
            pred = torch.multinomial(probs, 1)[0]
            token_list.append(pred)
            x = pred
        return token_list
class MultiHeadAttenion(nn.Module):
    def __init__(self, embed_dim:int, att_dim: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SingleHeadAttention(embed_dim, att_dim // num_heads))

    def forward(self, embed: Tensor) -> Tensor:
        out=[]
        for head in self.heads:
            out.append(head(embed))
        concat = torch.cat(out, dim=-1)
        return torch.round(concat, decimals=5)

class SingleHeadAttention(nn.Module):
    def __init(self, embed_dim, att_dim):
        super().__init__()
        self.keys = nn.Linear(embed_dim, att_dim)
        self.queries = nn.Linear(embed_dim, att_dim)
        self.values = nn.Linear(embed_dim, att_dim)

    def forward(self, embed: Tensor) -> Tensor:
        k = self.keys(embed)
        q = self.queries(embed)
        v = self.values(embed)

        dot_scores = q @ torch.transpose(k, -2, -1)
        B,T,C = k.shape
        dot_scores = dot_scores/(C**0.5)

        tril = torch.tril(torch.ones(T, T))
        dot_scores = dot_scores.masked_fill(tril == 0, float('-inf'))
        dot_scores = nn.functional.softmax(dot_scores, dim=-1)
        transform = dot_scores @ v
        return torch.round(transform, decimals=5)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.pos_encoding = torch.empty(max_seq_len, embed_size)

        position = torch.arange(max_seq_len).unsqueeze(dim=1)
        div_term = 10000**(-1/embed_size*torch.arange(0, embed_size, 2))
        leterm = position * div_term
        self.pos_encoding[:, 0::2] = torch.sin(leterm)
        self.pos_encoding[:, 1::2] = torch.cos(leterm)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_encoding[:x.size(0)].to(device)


if __name__ == '__main__':
    from io_utils import load_vocabulary

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load('trained_models/model.pth')
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    tl = model.generate_tokens(torch.tensor(vocab["there"], dtype=torch.int64), 40)
    for val in tl:
        print(vocab_rev[val.item()], end=" ")
