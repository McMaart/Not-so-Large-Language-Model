import unittest
import io_utils
import os


class TestIO(unittest.TestCase):
    def setUp(self):
        # f = open("DummyStories.txt", "w+")
        self.dummy_stories = ['This is a story.', 'This is also a story.', '"This is a quote in story 3".', 'test.']
        self.dummy_token_dict = {'this': 3, 'is': 3, 'a': 3, 'story': 3, '.': 4, 'also': 1, 'quote': 1, 'in': 1, '3': 1,
                                 'test': 1}
    def tearDown(self):
        pass

    def test_load_from_file(self):
        self.assertRaises(FileNotFoundError, io_utils.load_from_file, "non_existing.txt")

        loaded_stories = io_utils.load_from_file("DummyStories.txt")
        #print(loaded_stories)
        self.assertListEqual(self.dummy_stories, loaded_stories)
        self.assertEqual(4, len(loaded_stories))

    def test_get_vocabulary(self):
        created_dict = io_utils.get_vocabulary(self.dummy_stories)
        self.assertEqual(10, len(created_dict))
        #print(io_utils.get_vocabulary(self.dummy_stories))
        self.assertDictEqual(self.dummy_token_dict, created_dict)

    def test_save_to_file(self):
        test_file = io_utils.save_to_file("test.txt",['hi', 'hi2'])
        self.assertListEqual(['hi', 'hi2'], io_utils.load_from_file('test.txt'))
        if os.path.exists("test.txt"):
            os.remove("test.txt")

if __name__ == '__main__':
    unittest.main()
