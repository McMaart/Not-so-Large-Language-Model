import unittest
import io_utils
import os
from unittest.mock import patch, mock_open, MagicMock, call
import torch


class TestIO(unittest.TestCase):
    def setUp(self):
        self.dummy_stories = ['This is a story.', 'This is also a story.', '"This is a quote in story 3".', 'test.']
        self.dummy_token_dict = {'this': 3, 'is': 3, 'a': 3, 'story': 3, '.': 4, 'also': 1, 'quote': 1, 'in': 1, '3': 1,
                                 'test': 1}

    def tearDown(self):
        pass

    @patch('io_utils.load_dataset')
    def test_load_tiny_stories(self, mock_load_dataset):
        # Creating a MagicMock to handle the slicing
        mock_load_dataset.return_value = MagicMock()
        mock_load_dataset.return_value.__getitem__.return_value.__getitem__.return_value = {
            'text': ['This is a story.', 'This is also a story.']}
        result = io_utils.load_tiny_stories(end=2)
        self.assertListEqual(result, ['This is a story.', 'This is also a story.'])

    def test_load_from_file(self):
        self.assertRaises(FileNotFoundError, io_utils.load_from_file, "non_existing.txt")

        loaded_stories = io_utils.load_from_file("DummyStories.txt")
        #print(loaded_stories)
        self.assertListEqual(self.dummy_stories, loaded_stories)
        self.assertEqual(4, len(loaded_stories))

    def test_save_to_file(self):
        test_file = io_utils.save_to_file("test.txt",['hi', 'hi2'])
        self.assertListEqual(['hi', 'hi2'], io_utils.load_from_file('test.txt'))
        if os.path.exists("test.txt"):
            os.remove("test.txt")

    def test_get_vocabulary_frequencies(self):
        created_dict = io_utils.get_vocabulary_frequencies(self.dummy_stories)
        self.assertEqual(10, len(created_dict))
        #print(io_utils.get_vocabulary(self.dummy_stories))
        self.assertDictEqual(self.dummy_token_dict, created_dict)

    def test_get_vocabulary_idx(self):
        self.dummy_token_idx = {'this': 0, 'is': 1, 'a': 2, 'story': 3, '.': 4, 'also': 5, 'quote': 6, 'in': 7, '3': 8,
                                'test': 9}
        self.assertDictEqual(self.dummy_token_idx, io_utils.get_vocabulary_idx(self.dummy_stories))

    #Test for map_story_to_tensor
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
