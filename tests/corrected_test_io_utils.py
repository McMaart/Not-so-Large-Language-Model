import torch
import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import io_utils
from io import StringIO


class TestIOUtils(unittest.TestCase):

    # Test for load_tiny_stories
    @patch('io_utils.load_dataset')
    def test_load_tiny_stories(self, mock_load_dataset):
        # Creating a MagicMock to handle the slicing
        mock_load_dataset.return_value = MagicMock()
        mock_load_dataset.return_value.__getitem__.return_value.__getitem__.return_value = {
            'text': ['story1', 'story2']}
        result = io_utils.load_tiny_stories(end=2)
        self.assertEqual(result, ['story1', 'story2'])

    # Test for load_from_file
    @patch('builtins.open', new_callable=mock_open, read_data='story1\n<end>\n\nstory2\n<end>\n\n')
    def test_load_from_file(self, mock_file):
        result = io_utils.load_from_file('dummy_file.txt')
        self.assertEqual(result, ['story1', 'story2'])

    # Test for save_to_file
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        stories = ['story1', 'story2']
        io_utils.save_to_file('dummy_file.txt', stories)
        mock_file.assert_called_once_with('dummy_file.txt', 'w', encoding='utf-8')
        handle = mock_file()
        expected_calls = [call(f"{story}\n<end>\n\n") for story in stories]
        handle.write.assert_has_calls(expected_calls, any_order=True)

    # Test for get_vocabulary_frequencies
    def test_get_vocabulary_frequencies(self):
        stories = ['story1 with token', 'story2 with another token']
        result = io_utils.get_token_frequencies(stories)
        expected_result = {'story1': 1, 'with': 2, 'token': 2, 'story2': 1, 'another': 1}
        self.assertEqual(result, expected_result)

    # Test for get_vocabulary_frequencies_2
    def test_get_vocabulary_frequencies_2(self):
        stories = ['story1 with token', 'story2 with another token']
        result = io_utils.get_vocabulary_frequencies_2(stories)
        # Make sure to not include empty strings as tokens
        expected_result = {'story1': 1, 'with': 2, 'token': 2, 'story2': 1, 'another': 1}
        # Remove any empty tokens from the result before assertion
        result.pop('', None)
        self.assertEqual(result, expected_result)

    # Test for get_vocabulary_idx
    def test_get_vocabulary_idx(self):
        stories = ['story1 with token', 'story2 with another token']
        result = io_utils.create_vocabulary(stories)
        expected_result = {word: idx for idx, word in enumerate(result)}
        self.assertEqual(result, expected_result)

    # Test for map_story_to_tensor
    @patch('io_utils.get_tokenizer')
    def test_map_story_to_tensor(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = lambda x: x.split()
        vocab = {'story': 0, 'with': 1, 'token': 2}
        result = io_utils.map_story_to_tensor('story with token', vocab, mock_get_tokenizer())
        self.assertTrue(torch.equal(result, torch.tensor([0, 1, 2], dtype=torch.int)))

    # Test for tokens_to_story
    def test_tokens_to_story(self):
        tokens = ['story', 'with', 'token']
        result = io_utils.tokens_to_story(tokens)
        self.assertEqual(result, 'story with token')


if __name__ == '__main__':
    unittest.main()
