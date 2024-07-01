import pickle
import unittest
from unittest.mock import patch, mock_open
import torch
from torch import Tensor
import numpy as np
from datasets import Dataset, DatasetDict
from io_utils import (
    load_tiny_stories,
    load_from_file,
    save_to_file,
    get_token_frequencies,
    create_vocabulary,
    map_story_to_tensor,
    tokens_to_story,
    save_vocabulary,
    load_vocabulary,
    calculate_statistics
)

class TestIOUtils(unittest.TestCase):
    """
    Unit test class for the io_utils module.
    """

    @patch('io_utils.load_from_disk')
    def test_load_tiny_stories(self, mock_load_from_disk):
        """
        Test loading a subset of stories from the dataset.
        """
        mock_data = {
            "train": Dataset.from_dict({'text': ['story 1', 'story 2']}),
            "validation": Dataset.from_dict({'text': ['story 3', 'story 4']})
        }
        mock_load_from_disk.return_value = DatasetDict(mock_data)
        stories = load_tiny_stories(2, 0, "train")
        self.assertEqual(stories, ["story 1", "story 2"])

    @patch('builtins.open', new_callable=mock_open, read_data="story 1\n<end>\n\nstory 2\n<end>\n\n")
    def test_load_from_file(self, mock_file):
        """
        Test loading stories from a file.
        """
        stories = load_from_file("dummy_file.txt")
        self.assertEqual(stories, ["story 1", "story 2"])

    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        """
        Test saving stories to a file.
        """
        save_to_file("dummy_file.txt", ["story 1", "story 2"])
        handle = mock_file()
        handle.write.assert_any_call("story 1\n<end>\n\n")
        handle.write.assert_any_call("story 2\n<end>\n\n")

    def test_get_token_frequencies(self):
        """
        Test calculating token frequencies.
        """
        story_list = ["This is a test.", "This is another test."]
        tokenizer = lambda x: x.split()
        frequencies = get_token_frequencies(story_list, tokenizer)
        expected_frequencies = {'this': 2, 'is': 2, 'a': 1, 'test.': 2, 'another': 1}
        self.assertEqual(frequencies, expected_frequencies)

    def test_get_token_frequencies_empty(self):
        """
        Test token frequencies for an empty list.
        """
        story_list = []
        tokenizer = lambda x: x.split()
        frequencies = get_token_frequencies(story_list, tokenizer)
        expected_frequencies = {}
        self.assertEqual(frequencies, expected_frequencies)

    def test_create_vocabulary(self):
        """
        Test creating a vocabulary from stories.
        """
        story_list = ["This is a test.", "This is another test."]
        tokenizer = lambda x: x.split()
        vocab = create_vocabulary(story_list, tokenizer, max_words=8)
        expected_vocab = {'this': 0, 'is': 1, 'test.': 2, 'a': 3, 'another': 4, '<eos>': 5, '<unk>': 6, '<pad>': 7}
        self.assertEqual(vocab, expected_vocab)

    def test_map_story_to_tensor(self):
        """
        Test mapping a story to a tensor.
        """
        story = "This is a test."
        vocab = {'this': 0, 'is': 1, 'a': 2, 'test.': 3, '<unk>': 4}
        tokenizer = lambda x: x.split()
        tensor = map_story_to_tensor(story.lower(), vocab, tokenizer)
        expected_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        self.assertTrue(torch.equal(tensor, expected_tensor))

    def test_map_story_to_tensor_empty(self):
        """
        Test mapping an empty story to a tensor.
        """
        story = ""
        vocab = {'<unk>': 0}
        tokenizer = lambda x: x.split()
        tensor = map_story_to_tensor(story, vocab, tokenizer)
        expected_tensor = torch.tensor([], dtype=torch.int32)
        self.assertTrue(torch.equal(tensor, expected_tensor))

    def test_tokens_to_story(self):
        """
        Test converting tokens to a story.
        """
        tokens = ['this', 'is', 'a', 'test']
        story = tokens_to_story(tokens)
        self.assertEqual(story, 'This is a test')

    @patch('builtins.open', new_callable=mock_open)
    def test_save_vocabulary(self, mock_file):
        """
        Test saving the vocabulary to a file.
        """
        vocab = {'this': 0, 'is': 1, 'a': 2, 'test': 3}
        save_vocabulary(vocab, 'vocab.pkl')
        mock_file().write.assert_called()

    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps({'this': 0, 'is': 1, 'a': 2, 'test': 3}))
    def test_load_vocabulary(self, mock_file):
        """
        Test loading the vocabulary from a file.
        """
        vocab = load_vocabulary('vocab.pkl')
        self.assertEqual(vocab, {'this': 0, 'is': 1, 'a': 2, 'test': 3})

    def test_calculate_statistics(self):
        """
        Test calculating statistics from stories.
        """
        train_stories = ["This is a test.", "This is another test.", "This test is a test."]
        test_stories = ["A different test story.", "Yet another test story.", "Testing is fun."]

        # Combine train and test stories to simulate the entire dataset
        combined_stories = train_stories + test_stories

        tokenizer = lambda x: x.split()

        # Calculate statistics for the combined dataset
        total_tokens, unique_tokens, avg_seq_length, std_dev_seq_length, token_counts, lower_50_percent_max_frequency, lower_25_percent_max_frequency = calculate_statistics(
            combined_stories, tokenizer)

        print(f"Standard deviation calculated: {std_dev_seq_length}")
        print(f"Token counts: {token_counts}")

        self.assertEqual(total_tokens, 24)
        self.assertEqual(unique_tokens, 12)
        self.assertEqual(avg_seq_length, 24 / 6)

        # Corrected expected standard deviation based on the actual token counts
        expected_std_dev = np.std([4, 4, 5, 4, 4, 3], ddof=0)
        self.assertAlmostEqual(std_dev_seq_length, expected_std_dev)

        self.assertEqual(token_counts, [4, 4, 5, 4, 4, 3])

        # Max frequencies for the combined dataset
        self.assertEqual(lower_50_percent_max_frequency, 2)
        self.assertEqual(lower_25_percent_max_frequency, 1)

    if __name__ == '__main__':
        unittest.main()

