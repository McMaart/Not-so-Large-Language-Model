"""
RNN reference models
"""
import torch
from torch import nn, Tensor
from torch.nn.utils import rnn


class RNNBaseModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout2: float,
                 padding_idx: int | None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout2)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device,
                           requires_grad=False)

    def _get_packed_embedding(self, x: Tensor, lengths: Tensor | None = None):
        embedding = self.embedding(x)
        if lengths is None:
            return embedding
        else:
            return rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)

    def _get_forward_output(self, rnn_out: Tensor | rnn.PackedSequence, lengths: Tensor | None):
        if lengths is not None:
            rnn_out, _ = rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = self.dropout(rnn_out)
        return self.linear(rnn_out)


class RNNModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 512, hidden_size: int = 576, num_layers: int = 2,
                 dropout_rnn: float = 0.15, dropout2: float = 0.0, padding_idx: int | None = None):

        super().__init__(vocab_size, embed_size, hidden_size, num_layers, dropout2, padding_idx)
        self.rnn = nn.RNN(self.embed_size, self.hidden_size, self.num_layers, batch_first=True,
                          dropout=dropout_rnn if self.num_layers != 1 else 0.0)
        self.dropout = nn.Dropout(dropout2)

    def forward(self, x: Tensor, lengths: Tensor | None = None):
        h = self.init_hidden(x.size(0))
        embedding = self._get_packed_embedding(x, lengths)

        out, _ = self.rnn(embedding, h)
        return self._get_forward_output(out, lengths)


class GRUModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 384, hidden_size: int = 512, num_layers: int = 1,
                 dropout_gru: float = 0.05, dropout2: float = 0.0, padding_idx: int | None = None):

        super().__init__(vocab_size, embed_size, hidden_size, num_layers, dropout2, padding_idx)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first=True,
                          dropout=dropout_gru if self.num_layers != 1 else 0.0)
        self.dropout = nn.Dropout(dropout2)

    def forward(self, x: Tensor, lengths: Tensor | None = None):
        h = self.init_hidden(x.size(0))
        embedding = self._get_packed_embedding(x, lengths)

        out, _ = self.gru(embedding, h)
        return self._get_forward_output(out, lengths)


class LSTMModel(RNNBaseModel):
    def __init__(self, vocab_size: int, embed_size: int = 320, hidden_size: int = 512, num_layers: int = 1,
                 dropout_lstm: float = 0.1, dropout2: float = 0.0, padding_idx: int | None = None):

        super().__init__(vocab_size, embed_size, hidden_size, num_layers, dropout2, padding_idx)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True,
                            dropout=dropout_lstm if self.num_layers != 1 else 0.0)

    def forward(self, x: Tensor, lengths: Tensor | None = None):
        h, c = self.init_hidden(x.size(0)), self.init_hidden(x.size(0))
        embedding = self._get_packed_embedding(x, lengths)

        out, _ = self.lstm(embedding, (h, c))
        return self._get_forward_output(out, lengths)


if __name__ == "__main__":
    # from eval import calculate_rouge_scores, get_stories, max_rouge_score
    from io_utils import prompt_model

    prompt = "there"
    length = 100
    num_stories = 1

    story = prompt_model("rnn", prompt, length)
    print(story)
    # print(max_rouge_score(story, prompt))

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
