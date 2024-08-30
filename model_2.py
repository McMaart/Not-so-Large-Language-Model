"""
Implementation of the RNN (vanilla-RNN, GRU and LSTM) baseline models (ref. Subsection 3.4).
"""
import torch
from torch import nn, Tensor
from torch.nn.utils import rnn


class RNNBaseModel(nn.Module):
    """
    Superclass for implementing the RNN models.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout2: float,
                 padding_idx: int | None):
        """
        :param vocab_size: The number of tokens of the vocabulary.
        :param embed_size: The embedding dimension.
        :param hidden_size: The number of features for the hidden state.
        :param num_layers: The number of layers.
        :param dropout2: The dropout that is applied after the last RNN layer.
        :param padding_idx: The index of the padding token.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout2)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size: int) -> Tensor:
        """
        Initializes the hidden state of an RNN model.
        :param batch_size: The size of the batch.
        :return: The initialized hidden state of shape [num_layers, batch_size, hidden_size].
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=next(self.parameters()).device,
                           requires_grad=False)

    def _get_packed_embedding(self, x: Tensor, lengths: Tensor | None = None):
        """
        Returns the embedding for the input tensor.
        If the lengths of the sequences are provided, returns a PackedSequence.
        :param x: Input tensor (the token indices).
        :param lengths: The lengths of the sequences in the batch.
        :return: If lengths is None, returns the embedded sequence as Tensor. Else, a PackedSequence is returned.
        """
        embedding = self.embedding(x)
        if lengths is None:
            return embedding
        else:
            return rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)

    def _get_forward_output(self, rnn_out: Tensor | rnn.PackedSequence, lengths: Tensor | None):
        """
        Unpacks the RNN layer output (if necessary), and applies the dropout and linear layer.
        :param rnn_out: The output from the RNN (can be a Tensor or PackedSequence).
        :param lengths: Tensor containing the lengths of the sequences in the batch.
         Will be used for unpacking a PackedSequence.
        :return: The final output tensor of shape [batch_size, seq_len, vocab_size].
        """
        if lengths is not None:
            rnn_out, _ = rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = self.dropout(rnn_out)
        return self.linear(rnn_out)


# For the following classes, the input should be a Tensor of shape [batch_size, seq_len, embed_size]
# For more details, see implementation of the RNNBaseModel above.
class RNNModel(RNNBaseModel):
    """
    Implementation of the vanilla-RNN model.
    """
    def __init__(self, vocab_size: int, embed_size: int = 608, hidden_size: int = 1152, num_layers: int = 2,
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
    """
    Implementation of the GRU model.
    """
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
    """
    Implementation of the LSTM model.
    """
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
