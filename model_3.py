import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 32
max_seq_len = 256
num_special_non_eos_tokens = 2
num_special_tokens = 3

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_len = max_seq_len

    def forward(self, x):
        t = torch.arange(self.max_seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

    def apply_rotary_pos_emb(self, q, k, sincos):
        sin, cos = sincos[:, :, : self.dim], sincos[:, :, self.dim :]
        q1, q2 = q[..., : self.dim], q[..., self.dim :]
        k1, k2 = k[..., : self.dim], k[..., self.dim :]
        q = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
        k = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
        return q, k

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 512, nhead: int = 4, num_layers: int = 4, dim_ff: int = 2048,
                 dropout: float = 0.1, padding_idx: int | None = None, pos_enc_type: str = 'sinusoidal'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.nhead = nhead
        self.pos_enc_type = pos_enc_type

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)

        if pos_enc_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(embed_size, base=10000)
        elif pos_enc_type == 'rope':
            self.pos_encoding = RotaryPositionalEncoding(dim=embed_size // nhead, max_seq_len=max_seq_len, base=10000)

        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                                                   batch_first=True, activation="gelu", norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        embedding: Tensor = self.embedding(x)

        if self.pos_enc_type == 'rope':
            sincos = self.pos_encoding(embedding)
            embedding = embedding.view(embedding.size(0), embedding.size(1), self.nhead, -1)
            sincos = sincos[:, :embedding.size(1), :]
        else:
            embedding = self.pos_encoding(embedding)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=torch.bool, device=torch.device(device))
        if lengths is None:
            pad_mask = None
        else:
            lengths = lengths.to(device, non_blocking=True)
            seq_indices = torch.arange(x.size(1), device=device)
            pad_mask = seq_indices >= lengths[:, None]

        if self.pos_enc_type == 'rope':
            for layer in self.encoder.layers:
                layer.self_attn = self.apply_rope_to_attention(layer.self_attn, sincos)
            embedding = self.encoder(embedding, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        else:
            embedding = self.encoder(embedding, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)

        return self.linear(embedding)

    def apply_rope_to_attention(self, self_attn, sincos):
        original_forward = self_attn.forward

        def new_forward(query, key, value, *args, **kwargs):
            # Ensure query and key are reshaped correctly for RoPE
            batch_size, seq_len, embed_size = query.size()
            query = query.view(batch_size, seq_len, self.nhead, embed_size // self.nhead)
            key = key.view(batch_size, seq_len, self.nhead, embed_size // self.nhead)

            # Apply rotary positional embeddings
            query, key = self.pos_encoding.apply_rotary_pos_emb(query, key, sincos)

            # Reshape back to original dimensions
            query = query.view(batch_size, seq_len, embed_size)
            key = key.view(batch_size, seq_len, embed_size)

            return original_forward(query, key, value, *args, **kwargs)

        self_attn.forward = new_forward
        return self_attn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, dropout: float = 0.07, base: int = 10000):
        super().__init__()
        pos_enc = torch.empty(max_seq_len, embed_size, dtype=torch.float)
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        inv_denominator = base ** (-1 / embed_size * torch.arange(0, embed_size, 2, dtype=torch.float))
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

# Example usage (for illustration):
if __name__ == '__main__':
    vocab_size = 10000
    embed_size = 512
    nhead = 8
    num_layers = 6
    model = TransformerModel(vocab_size=vocab_size, embed_size=embed_size, nhead=nhead, num_layers=num_layers, pos_enc_type='rope')
    print(model)
