import unittest
from unittest.mock import patch, mock_open
from io_utils import (
    get_token_frequencies,
    create_vocabulary,
    tokens_to_story,
    save_vocabulary,
)


class TestIOUtils(unittest.TestCase):
    """
    Unit test class for the io_utils module.
    """

    def test_get_token_frequencies(self):
        """
        Test calculating token frequencies.
        """
        story_list = ["This is a test.", "This is another test."]
        tokenizer = lambda x: x.lower().split()
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
        tokenizer = lambda x: x.lower().split()
        vocab = create_vocabulary(story_list, tokenizer, max_words=9)

        # Check that the expected keys exist in the vocabulary, including 'another'
        expected_vocab_keys = {'this', 'is', 'test.', 'a', 'another', '<eos>', '<bos>', '<unk>', '<pad>'}
        self.assertEqual(set(vocab.keys()), expected_vocab_keys)

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

    """
    Test for load vocabulary with mock object.
    
    @patch('io_utils.get_absolute_path', return_value='dummy_path.pkl')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_vocabulary(self, mock_open, mock_get_path):
        
        Test loading the vocabulary from a file.
        
        # Mock the file content to return expected pickle data
        mock_open.return_value.read.return_value = pickle.dumps({'this': 0, 'is': 1, 'a': 2, 'test': 3})

        # Execute the function that loads the vocabulary
        vocab = load_vocabulary('dummy_path.pkl')

        # Check that the loaded vocabulary matches expected data
        self.assertEqual(vocab, {'this': 0, 'is': 1, 'a': 2, 'test': 3})
    """

if __name__ == '__main__':
    unittest.main()
