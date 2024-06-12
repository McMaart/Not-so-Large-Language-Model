"""
RNN reference model
"""
import torch
from torch import nn, Tensor


class RNNBaseModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, padding_idx: int | None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device,
                           requires_grad=False)


class RNNModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 384, num_layers: int = 6,
                 dropout: float = 0.1, padding_idx: int | None = None, dropout2: float = 0.1):
        super().__init__(vocab_size, embed_size, hidden_size, num_layers, padding_idx)
        self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout2)

    def forward(self, x: Tensor):
        h = self.init_hidden(x.size(0))
        embedding = self.embedding(x)
        out, _ = self.rnn(embedding, h)
        out = self.dropout(out)
        return self.linear(out)


class GRUModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, padding_idx: int | None = None):
        super().__init__(vocab_size, embed_size, hidden_size, num_layers, padding_idx)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)

    def forward(self, x: Tensor):
        h = self.init_hidden(x.size(0))
        embedding = self.embedding(x)
        out, _ = self.gru(embedding, h)
        return self.linear(out)


class LSTMModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 256, hidden_size: int = 256, num_layers: int = 4,
                 dropout_lstm: float = 0.2, padding_idx: int | None = None):
        super().__init__(vocab_size, embed_size, hidden_size, num_layers, padding_idx)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout_lstm)

    def forward(self, x: Tensor):
        h, c = self.init_hidden(x.size(0)), self.init_hidden(x.size(0))
        embedding = self.embedding(x)
        out, _ = self.lstm(embedding, (h, c))
        return self.linear(out)


if __name__ == "__main__":
    from eval import calculate_rouge_scores, get_stories, max_rouge_score
    from io_utils import prompt_model

    prompt = "there"
    length = 100
    num_stories = 1
    
    story = prompt_model("RNN_256_2_128_run_1_32_0.0008_16473.354659267", prompt, length)
    print(story)
    print()
    print(max_rouge_score(story, prompt))

    # stories = [prompt_model("rnn_model", prompt, length) for _ in range(num_stories)]
    # for story in stories:
    #     print(story)
    #     print()
    
    # actual_stories = get_stories(prompt, 10000)
    # print(len(stories))
    # print(len(actual_stories))

    # scores = calculate_rouge_scores(stories, actual_stories[:num_stories])
    # print(scores)
    # print(f"ROUGE-1: {scores['rouge-1']}")
    # print(f"ROUGE-2: {scores['rouge-2']}")
    # print(f"ROUGE-L: {scores['rouge-l']}")
    # print(f"ROUGE-L: {scores['rouge-l']}")

