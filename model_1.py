import torch
from torch import nn, Tensor
import torch.nn.functional as F
import packaging
from torchtune.modules import RotaryPositionalEmbeddings


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate = 0.01
batch_size = 128
max_seq_len = 256
num_special_non_eos_tokens = 2
num_special_tokens = 3


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
            self.pos_encoding = RotaryPositionalEmbeddings(dim=embed_size // nhead, max_seq_len=max_seq_len, base=10000)

        # with flash attention
        #self.encoder_layers = nn.ModuleList([
           # FlashMHA(embed_size, nhead, p_dropout=dropout, causal=True) for _ in range(num_layers)
        #])

        #without flash attention
        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                                                   batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
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

        # with flash attention
        #for layer in self.encoder_layers:
           # embedding, _ = layer(embedding, attn_mask=mask)

        # without flash attention
        embedding = self.encoder(embedding, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        return self.linear(embedding)


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


@torch.no_grad()
def generate_tokens(model: nn.Module, token_tensor: Tensor, length: int = 250, temperature: float = 1.0,
                    eos_token: int = None) -> Tensor:
    model.eval()
    for _ in range(len(token_tensor[0]), length + 1):
        output = model(token_tensor)[:, -1, :-num_special_non_eos_tokens]
        if abs(temperature) < 1e-10:
            pred = torch.argmax(output, dim=1).unsqueeze(0)
        else:
            probs = F.softmax(output * (1 / temperature), dim=-1)
            pred = torch.multinomial(probs, 1)
        token_tensor = torch.cat((token_tensor, pred), 1)
        if pred.item() == eos_token:
            return token_tensor
    return token_tensor


@torch.no_grad()
def generate_tokens_beam(model: nn.Module, input_tensor: Tensor, beam_width: int, length: int = 250,
                         temperature: float = 1.0, eos_token: int = None) -> Tensor:
    model.eval()
    sequences = [(input_tensor.squeeze(0).tolist(), 0.0)]  # List of sequences with their scores
    completed_sequences = []

    for _ in range(length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == eos_token or len(seq) >= length:  # Cap sequence length
                completed_sequences.append((seq, score))
                continue

            input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            output = model(input_tensor)[:, -1, :-num_special_non_eos_tokens]

            if abs(temperature) < 1e-10:
                # Temperature = 0, select the max logit deterministically
                logits = output
                pred = torch.argmax(logits, dim=-1).item()
                log_prob = F.log_softmax(logits, dim=-1).squeeze(0)[pred].item()
                candidate = (seq + [pred], score + log_prob)
                all_candidates.append(candidate)
            else:
                logits = output / temperature
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width)
                for j in range(beam_width):
                    candidate = (seq + [top_k_tokens[j].item()], score + top_k_log_probs[j])
                    all_candidates.append(candidate)

        if len(all_candidates) == 0:
            break

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        if all(seq[-1] == eos_token or len(seq) >= length for seq, _ in sequences):
            break

    best_sequence = max(completed_sequences + sequences, key=lambda tup: tup[1])[0]
    return torch.tensor(best_sequence).unsqueeze(0)


@torch.no_grad()
def generate_tokens_beam_multinomial(model: nn.Module, input_tensor: Tensor, beam_width: int, length: int = 250,
                                     temperature: float = 1.0, eos_token: int = None, top_k: int = 50) -> Tensor:
    model.eval()
    sequences = [[(input_tensor.squeeze(0).tolist(), 0.0)]]
    completed_sequences = []

    for step in range(length):
        all_candidates = []

        for beam in sequences:
            for seq, score in beam:
                if seq[-1] == eos_token or len(seq) >= length:  # Cap sequence length
                    completed_sequences.append((seq, score))
                    continue

                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                output = model(input_tensor)[:, -1, :-num_special_non_eos_tokens]

                if abs(temperature) < 1e-10:
                    logits = output
                    pred = torch.argmax(logits, dim=-1).item()
                    log_prob = F.log_softmax(logits, dim=-1).squeeze(0)[pred].item()
                    candidate = (seq + [pred], score + log_prob)
                else:
                    logits = output / temperature
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    top_k_log_probs, top_k_tokens = torch.topk(log_probs, top_k)
                    top_k_probs = F.softmax(top_k_log_probs, dim=-1)
                    pred_idx = torch.multinomial(top_k_probs, 1).item()
                    pred = top_k_tokens[pred_idx].item()
                    candidate = (seq + [pred], score + top_k_log_probs[pred_idx].item())

                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = [ordered[i:i + beam_width] for i in range(0, len(ordered), beam_width)]

        if all(seq[-1] == eos_token or len(seq) >= length for beam in sequences for seq, _ in beam):  # Cap sequence length
            break

    best_sequence = max(completed_sequences, key=lambda tup: tup[1])[0] if completed_sequences else sequences[0][0][0]
    return torch.tensor(best_sequence).unsqueeze(0)




if __name__ == '__main__':
    from io_utils import prompt_model

    string = '"What do birds like to eat?", Tom asked his mother.'
    #string = 'Once'
    story = prompt_model("10M", string, 255, 0, '', beam_width=8)
    print(story)
