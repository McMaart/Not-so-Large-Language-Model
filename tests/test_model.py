import unittest
import torch
from model_1 import TransformerModel, device


class TestTransformerModelDims(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 2048
        self.embed_size = 128
        self.model = TransformerModel(self.vocab_size, self.embed_size).to(device)

    def test_output_shape(self):
        """
        Asserts that the output shape of the model is [batch_size, seq_len, vocab_size]
        """
        test_tuples = [(1, 1), (16, 24)]
        for batch_size, seq_len in test_tuples:
            a = torch.randint(high=self.vocab_size - 1, size=(batch_size, seq_len), dtype=torch.int32, device=device)
            self.assertEqual(self.model(a).shape, torch.Size([batch_size, seq_len, self.vocab_size]))

    def test_layers_shape(self):
        """
         Asserts that the output shape of the individual layers are correct
         """
        # Test output shape of the embedding layer
        test_tuples_embedding = [(1, 1), (16, 24)]
        for batch_size, seq_len in test_tuples_embedding:
            a = torch.randint(high=self.vocab_size - 1, size=(batch_size, seq_len), dtype=torch.int32, device=device)
            self.assertEqual(self.model.embedding(a).shape, torch.Size([batch_size, seq_len, self.embed_size]))

        # Test output shape of the positional encoding, encoder and linear layer
        test_tuples_layers = [(1, 1, self.embed_size), (16, 24, self.embed_size)]
        for batch_size, seq_len, embed_size in test_tuples_layers:
            a = torch.rand(size=(batch_size, seq_len, embed_size), dtype=torch.float, device=device) * 1337.
            for layer in (self.model.pos_encoding, self.model.encoder):
                self.assertEqual(layer(a).shape, torch.Size([batch_size, seq_len, self.embed_size]))
            self.assertEqual(self.model.linear(a).shape, torch.Size([batch_size, seq_len, self.vocab_size]))

    def test_pos_enc(self):
        """
        Asserts that the positional coding only contains values between -1 and 1
        (since we only use sine and cosine functions for our positional encoding)
        """
        pe = self.model.pos_encoding.pos_enc
        eps = 1e-12
        self.assertTrue(torch.max(torch.abs(pe)) <= 1.0 + eps)


if __name__ == '__main__':
    unittest.main()
