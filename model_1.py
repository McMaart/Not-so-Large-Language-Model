import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 0.00012381301478401363
batch_size = 64
max_seq_len = 256
num_special_tokens = 2


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 512, nhead: int = 4, num_layers: int = 4, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)

        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                                                   batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding: Tensor = self.embedding(x)
        embedding = self.pos_encoding(embedding)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        embedding = self.encoder(embedding, mask=mask.to(device), is_causal=True)
        return self.linear(embedding)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor, length: int) -> Tensor:
        self.eval()
        token_tensor = start_token
        for _ in range(length):
            pred1 = self(token_tensor)[:, -1, :-num_special_tokens]
            probs = F.softmax(pred1, dim=-1)
            pred = torch.multinomial(probs, 1)
            token_tensor = torch.cat((token_tensor, pred), 1)
        return token_tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.07):
        super().__init__()
        pos_enc = torch.empty(max_seq_len, embed_size, dtype=torch.float)
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        inv_denominator = 10000 ** (-1 / embed_size * torch.arange(0, embed_size, 2, dtype=torch.float))
        pe_term = pos * inv_denominator
        pos_enc[:, 0::2] = torch.sin(pe_term)
        pos_enc[:, 1::2] = torch.cos(pe_term)

        # Register as buffer so the positional encoding is not passed to the optimizer
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the positional encoding (and dropout) to the embedding.
        :param x: Tensor of shape [batch_size, seq_len, embed_size]
        :return: Tensor of shape [batch_size, seq_len, embed_size]
        """
        return self.dropout(x + self.pos_enc[:x.size(1)])


if __name__ == '__main__':
    from io_utils import prompt_model

    story = prompt_model("model", "once", 250)
    print(f"\n{story}")
