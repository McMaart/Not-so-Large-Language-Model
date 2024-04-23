from torch import nn, Tensor


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.embedding(x)
        return self.linear(embedding)
