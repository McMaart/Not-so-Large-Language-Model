"""
Tests for asserting that the output dimensions of the Transformer and RNN models are correct.
"""
import unittest
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from models.model_1 import TransformerModel, device, PositionalEncoding
from models.model_2 import RNNModel, LSTMModel, GRUModel


def generate_batch_with_lengths(batch_size: int, vocab_size: int, max_seq_len: int) -> tuple[Tensor, Tensor]:
    """
    Generates a batch of random sequences with varying lengths.
    :param batch_size: The batch size (i.e, the number of sequences).
    :param vocab_size: The number of tokens in the vocabulary.
    :param max_seq_len: The maximum length of a sequence in the batch.
    :return: A tuple containing
        - The padded input Tensor of shape [batch_size, max_seq_len]
        - The length of the individual sequences (Tensor of shape [batch_size]).
    """
    seq_lengths = torch.randint(1, max_seq_len, size=(batch_size,))
    sequences = [torch.randint(high=vocab_size - 2, size=(length,), dtype=torch.int) for length in seq_lengths]
    input_tensor = pad_sequence(sequences, batch_first=True, padding_value=vocab_size - 1).to(device)
    return input_tensor, seq_lengths


class TestTransformerModelDims(unittest.TestCase):
    """
    Tests for asserting that the output dimensions of the Transformer model (and its components) are correct.
    """
    def setUp(self):
        self.vocab_size = 2048
        self.embed_size = 128
        self.max_seq_len = 256
        self.test_tuples = [(1, 1), (16, 24), (32, 256)]  # corresponds to (batch_size, seq_len)
        self.model = TransformerModel(self.vocab_size, self.embed_size).to(device)

    def test_output_shape(self):
        """
        Asserts that the output shape of the model is [batch_size, seq_len, vocab_size].
        """
        for batch_size, seq_len in self.test_tuples:
            a = torch.randint(high=self.vocab_size - 1, size=(batch_size, seq_len), dtype=torch.int32, device=device)
            self.assertEqual(self.model(a).shape, torch.Size((batch_size, seq_len, self.vocab_size)))

    def test_layers_shape(self):
        """
         Asserts that the output shape of the individual layers are correct (for inputs with random values).
         """
        for batch_size, seq_len in self.test_tuples:
            # Test output shape of the embedding layer
            output_shape = (batch_size, seq_len, self.embed_size)
            a = torch.randint(high=self.vocab_size - 2, size=(batch_size, seq_len), dtype=torch.int32, device=device)
            self.assertEqual(self.model.embedding(a).shape, torch.Size(output_shape))

            # Test output shape of the positional encoding, encoder and linear layer
            a = torch.rand(size=output_shape, dtype=torch.float, device=device) * 42.0
            for layer in (self.model.pos_encoding, self.model.encoder):
                self.assertEqual(layer(a).shape, torch.Size(output_shape))
            self.assertEqual(self.model.linear(a).shape, torch.Size([batch_size, seq_len, self.vocab_size]))

    def test_model_with_lengths(self):
        """
        Asserts that the output shape of the model is [batch_size, seq_len, vocab_size] when provided with the
        lengths of the sequences in the batch.
        """
        for batch_size, _ in self.test_tuples:
            input_tensor, seq_lengths = generate_batch_with_lengths(batch_size, self.vocab_size, self.max_seq_len)
            pred = self.model(input_tensor, seq_lengths)
            self.assertEqual(pred.shape, torch.Size((*input_tensor.shape, self.vocab_size)))

    def test_sin_pos_enc(self, eps: float = 1e-12):
        """
        Asserts that the sinusoidal positional coding only contains values between -1 and 1.
        """
        pe = PositionalEncoding(self.embed_size)
        self.assertTrue(torch.max(torch.abs(pe.pos_enc)) <= 1.0 + eps)


class TestRNNModelDims(unittest.TestCase):
    """
    Tests for asserting that the output dimensions of the RNN, GRU and LSTM model are correct.
    """
    def setUp(self):
        self.vocab_size = 2048
        self.embed_size = 128
        self.hidden_size = 256
        self.n_layers = 2
        self.max_seq_len = 256

        self.rnn_model = RNNModel(self.vocab_size, self.embed_size, self.hidden_size, self.n_layers).to(device)
        self.gru_model = GRUModel(self.vocab_size, self.embed_size, self.hidden_size, self.n_layers).to(device)
        self.lstm_model = LSTMModel(self.vocab_size, self.embed_size, self.hidden_size, self.n_layers).to(device)
        self.model_list = (self.rnn_model, self.gru_model, self.lstm_model)

        self.test_tuples = [(1, 1), (16, 24), (32, 256)]  # corresponds to (batch_size, seq_len)

    def test_output_shape(self):
        """
        Asserts that the output shape of the models is [batch_size, seq_len, vocab_size] (when no lengths are given).
        """
        for model in self.model_list:
            for batch_size, seq_len in self.test_tuples:
                a = torch.randint(high=self.vocab_size - 2, size=(batch_size, seq_len), dtype=torch.int32,
                                  device=device)
                self.assertEqual(model(a).shape, torch.Size((batch_size, seq_len, self.vocab_size)))

    def test_model_with_lengths(self):
        """
        Asserts that the output shape of the models is [batch_size, seq_len, vocab_size] when provided with the
        lengths of the sequences in the batch.
        """
        for batch_size, _ in self.test_tuples:
            input_tensor, seq_lengths = generate_batch_with_lengths(batch_size, self.vocab_size, self.max_seq_len)
            for model in self.model_list:
                pred = model(input_tensor, seq_lengths)
                print(pred.shape)
                self.assertEqual(pred.shape, torch.Size((*input_tensor.shape, self.vocab_size)))


if __name__ == '__main__':
    unittest.main()
