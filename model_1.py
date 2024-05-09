import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 5e-4
batch_size = 16
max_seq_len = 256
num_special_tokens = 1


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding: Tensor = self.embedding(x)
        embedding = self.pos_encoding(embedding)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0))
        attn_output = self.encoder(embedding, mask=mask, is_causal=True)
        return self.linear(attn_output)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int) -> list:
        self.eval()
        x = start_token
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1)
            pred = torch.multinomial(probs[:-num_special_tokens], 1)[0]  # The last token is "<unk>"
            token_list.append(pred)
            x = pred
        return token_list


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.07):
        super().__init__()
        pos_enc = torch.empty(max_seq_len, embed_size, dtype=torch.float)
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_seq_len, dtype=torch.float, device=device).unsqueeze(dim=1)
        inv_denominator = 10000 ** (-1 / embed_size * torch.arange(0, embed_size, 2, dtype=torch.float))
        pe_term = pos * inv_denominator
        self.pos_encoding[:, 0::2] = torch.sin(pe_term)
        self.pos_encoding[:, 1::2] = torch.cos(pe_term)

        # Register as buffer so the positional encoding is not passed to the optimizer
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])


if __name__ == '__main__':
    from io_utils import prompt_model

    story = prompt_model("model", "there", 42)
    print(f"\n{story}")
