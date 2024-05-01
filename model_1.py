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

    def generate_tokens(self, start_token: Tensor|int, length: int) -> list:
        x = start_token
        token_list = [x]
        for _ in range(length):
            probs = F.softmax(self(x), dim=-1)
            pred = torch.multinomial(probs, 1)[0]
            token_list.append(pred)
            x = pred
        return token_list


if __name__ == '__main__':
    from io_utils import load_vocabulary, tokens_to_story

    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load('trained_models/model.pth')
    except FileNotFoundError:
        model = TransformerModel(len(vocab))

    tl = model.generate_tokens(torch.tensor(vocab["there"], dtype=torch.int64), 40)
    token_list = []
    for val in tl:
        # print(vocab_rev[val.item()], end=" ")
        token_list.append(vocab_rev[val.item()])
    print("\n",tokens_to_story(token_list))
