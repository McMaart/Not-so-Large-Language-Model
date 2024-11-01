"""
Implementation of the Transformer model and positional encoding (ref. Subsection 3.3 in our project report).
"""
import torch
from torch import nn, Tensor
from torchtune.modules import RotaryPositionalEmbeddings


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
max_seq_len = 256
num_special_tokens = 4
num_special_non_eos_tokens = 3


class TransformerModel(nn.Module):
    """
    Implementation of the Transformer model.
    """
    def __init__(self, vocab_size: int, embed_size: int = 128, nhead: int = 4, num_layers: int = 4, dim_ff: int = 512,
                 dropout: float = 0.0, padding_idx: int | None = None, pos_enc_type: str = 'sinusoidal'):
        """
        :param vocab_size: The number of tokens in the vocabulary.
        :param embed_size: The embedding dimension.
        :param nhead: The number of heads in the multi-head attention mechanism.
        :param num_layers: The number of layers of the Transformer.
        :param dim_ff: The dimension of the feedforward layer.
        :param dropout: The rate of dropout (applied after adding the positional encoding to the embedding).
        :param padding_idx: The index of the padding token.
        :param pos_enc_type: The type of the positional encoding used. Options are 'sinusoidal' and 'rope'.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.nhead = nhead
        self.pos_enc_type = pos_enc_type

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        if pos_enc_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(embed_size, base=10_000)
        elif pos_enc_type == 'rope':
            self.pos_encoding = RotaryPositionalEmbeddings(dim=embed_size // nhead, max_seq_len=max_seq_len,
                                                           base=10_000)

        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                                                   batch_first=True, activation="gelu", norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Forward pass of the Transformer model.
        :param x: Input Tensor of shape [batch_size, seq_len].
        :param lengths: Lengths of the individual stories in the batch. Shape: [batch_size].
        :return: Output tensor of shape [batch_size, seq_len, vocab_size].
        """
        embedding: Tensor = self.embedding(x)
        if self.pos_enc_type == 'rope':
            embedding = embedding.view(embedding.size(0), embedding.size(1), self.nhead, -1)  # Reshape for multi-head attention
            embedding = self.pos_encoding(embedding)  # Apply ROPE
            embedding = embedding.view(embedding.size(0), embedding.size(1), -1)  # Reshape back to original
        else:
            embedding = self.pos_encoding(embedding)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), dtype=torch.bool, device=torch.device(device))
        if lengths is None:
            pad_mask = None
        else:
            lengths = lengths.to(device, non_blocking=True)
            seq_indices = torch.arange(x.size(1), device=device)
            pad_mask = seq_indices >= lengths[:, None]

        embedding = self.encoder(embedding, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        return self.linear(embedding)


class PositionalEncoding(nn.Module):
    """
    Implementation of the Sinusoidal Positional Encoding, proposed by Vaswani et al. ("Attention Is All You Need").
    """
    def __init__(self, embed_size: int, dropout: float = 0.0, base: int = 10_000):
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
