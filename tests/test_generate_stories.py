import unittest
import torch
from torch import nn
from generate_stories import generate_tokens, generate_tokens_beam, generate_tokens_beam_multinomial


class MockModel(nn.Module):
    def __init__(self, vocab_size):
        super(MockModel, self).__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(10, vocab_size)  # Simple linear layer

    def forward(self, x):
        batch_size, seq_len = x.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)


class TestGenerateStories(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.mock_model = MockModel(self.vocab_size)
        self.token_tensor = torch.randint(0, self.vocab_size, (1, 5))  # Random initial tokens
        self.eos_token = 99  # End-of-sequence token

    def test_generate_tokens(self):
        # Test basic token generation
        result = generate_tokens(self.mock_model, self.token_tensor, length=10, eos_token=self.eos_token)
        self.assertTrue(result.shape[1] == 10 or result.shape[1] == 11)

    def test_generate_tokens_with_eos(self):
        # Test token generation with early stopping at eos_token
        result = generate_tokens(self.mock_model, self.token_tensor, length=10, eos_token=self.eos_token)
        self.assertTrue(result[0, -1].item() == self.eos_token or result.shape[1] <= 11)

    def test_generate_tokens_beam(self):
        # Test beam search token generation
        result = generate_tokens_beam(self.mock_model, self.token_tensor, beam_width=3, length=10,
                                      eos_token=self.eos_token)
        self.assertTrue(result.shape[1] <= 10)

    def test_generate_tokens_beam_multinomial(self):
        # Test beam search with multinomial sampling
        result = generate_tokens_beam_multinomial(self.mock_model, self.token_tensor, beam_width=3, length=10,
                                                  eos_token=self.eos_token, top_k=5)
        self.assertTrue(result.shape[1] <= 10)


if __name__ == '__main__':
    unittest.main()
