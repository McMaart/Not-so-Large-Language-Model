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
        #self.mha = MultiHeadAttenion(model_dim, num_heads)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)
        self.to(device)
    def forward(self, x: Tensor) -> Tensor:
        #Änderung mit Batches
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = self.pos_encoding(x)  # Add positional encoding
        return self.linear(x)

        #embedding: Tensor = self.embedding(x).to(device)
        #embedding = self.pos_encoding(embedding).to(device)
        #return self.linear(embedding).to(device)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:
        self.eval()
        x = start_token.to(device)
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1).to(device)
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
    def __init__(self, embed_size: int, dropout: float = 0.07):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings once in the right shape.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pos_encoding = torch.zeros(max_seq_len, embed_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encodings as a buffer correctly
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, shape [batch_size, seq_len, embed_size]
        """
        # Use the registered buffer, ensuring we only apply as many positions as the current x's sequence length
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

        # Code vorher
        #pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        #inv_denominator = 10000**(-1/embed_size*torch.arange(0, embed_size, 2, dtype=torch.float))
        #pe_term = pos * inv_denominator
        #self.pos_encoding[:, 0::2] = torch.sin(pe_term)
        #self.pos_encoding[:, 1::2] = torch.cos(pe_term)

    #def forward(self, x: Tensor) -> Tensor:
        #return self.dropout(x + self.pos_encoding[:x.size(0)]).to(device)


if __name__ == '__main__':
    from io_utils import load_vocabulary

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load('trained_models/model.pth').to(device)
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    input_tensor = torch.tensor(vocab["once"], dtype=torch.int64).unsqueeze(0)
    tl = model.generate_tokens(input_tensor, 60)
    for val in tl:
        print(vocab_rev[val.item()], end=" ")


