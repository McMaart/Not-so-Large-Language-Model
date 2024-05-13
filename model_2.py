"""  
RNN reference model 
"""
import torch
from torch import nn, Tensor
from io_utils import prompt_model

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
learning_rate_rnn = 1e-3
batch_size_rnn = 64
max_seq_len_rnn = 60


class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 128, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.to(device)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        if h is None:
            if x.dim() == 2:  # Batched input
                h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            else:  # Unbatched input
                h = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.linear(out)
        return out, h

    def init_hidden(self, batch_size: int) -> Tensor:
      if batch_size == 1:  # Unbatched input
        return torch.zeros(self.num_layers, self.hidden_size).to(device)
      else:  # Batched input
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    @torch.no_grad()
    def generate_tokens(self, start_token: Tensor | int, length: int, eos_idx: int = None) -> list:
        self.eval()
        x = start_token.to(device)
        h = self.init_hidden(1)
        token_list = [x]
        for _ in range(length):
            out, h = self(x.unsqueeze(0), h)
            out = out.squeeze(0)
            x = torch.argmax(out, dim=-1)
            token_list.append(x)
            if eos_idx is not None and x == eos_idx:
                break
        return token_list




if __name__ == "__main__":
    print(prompt_model("rnn_model", "my", 100, end_on_eos=False))

