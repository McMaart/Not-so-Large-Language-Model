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
max_seq_len = 384


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size).to(device)
        self.pos_encoding = PositionalEncoding(embed_size).to(device)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead=8).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
        self.linear = nn.Linear(self.embed_size, self.vocab_size).to(device)
        #self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        embedding: Tensor = self.embedding(x).to(device)
        embedding = self.pos_encoding(embedding).to(device)
        src_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(device)
        embedding = self.encoder(embedding, mask=src_mask, is_causal=True)
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
        self.dropout = nn.Dropout(p=dropout).to(device)

        pos = torch.arange(max_seq_len, dtype=torch.float, device= device).unsqueeze(dim=1)
        inv_denominator = 10000**(-1/embed_size*torch.arange(0, embed_size, 2, dtype=torch.float)).to(device)
        pe_term = pos * inv_denominator
        self.pos_encoding[:, 0::2] = torch.sin(pe_term)
        self.pos_encoding[:, 1::2] = torch.cos(pe_term)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        return self.dropout(x + self.pos_encoding[:x.size(0)].to(device)).to(device)


if __name__ == '__main__':
    from io_utils import prompt_model, load_vocabulary

    # story = prompt_model("model", "there", 40)
    # print("\n", story)

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load('trained_models/model.pth')
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    input_tensor = torch.tensor(vocab["there"], dtype=torch.int64).unsqueeze(0)
    tl = model.generate_tokens(input_tensor, 42)
    for val in tl:
        print(vocab_rev[val.item()], end=" ")
