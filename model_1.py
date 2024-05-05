import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.embedding(x)
        return self.linear(embedding)

    def generate_tokens(self, start_token: Tensor|int, length: int, eos_idx: int = None) -> list:
        x = start_token
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1)
            pred = torch.multinomial(probs, 1)[0]
            token_list.append(pred)
            x = pred
            if eos_idx is not None and pred == eos_idx:
                break
        return token_list


if __name__ == '__main__':
    from io_utils import prompt_model
    story = prompt_model("model", "there", 400, end_on_eos=True)
    print("\n", story)
