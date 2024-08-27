import unittest
from data.preprocess_dataset import (
    find_non_ascii_symbols,
    clean_dataset,
    replacement_table,
    allowed_non_ascii_symbols
)


class TestPreprocessDataset(unittest.TestCase):
    def setUp(self):
        self.raw_stories = [
            "This is a test story. It has some characters like â€™ and â€“ which need to be replaced.",
            "Another storyâ€¦ with more â€œoddâ€ characters. Also with   multiple   spaces and *stars*.",
            "Another story with   multiple   spaces and *stars*   to remove. This is also longer than 50.",
            "Short",  # This should be removed if min_length is set to a value greater than its length.
            "Normal story with no issues."
        ]
        self.cleaned_stories_min_length_6 = [
            "Another story with multiple spaces and stars to remove. This is also longer than 50.",
            "Normal story with no issues."
        ]
        self.cleaned_stories_min_length_50 = ["Another story with multiple spaces and stars to remove. This is also longer than 50."
        ]
        self.min_length = 6

    def test_find_non_ascii_symbols(self):
        story_with_non_ascii = "This is a test story with non-ascii character â."
        story_with_allowed_non_ascii = (f"This is a test story with allowed non-ascii character "
                                        f"{" ".join(allowed_non_ascii_symbols)}.")
        story_without_non_ascii = "This is a plain ascii story."

        self.assertTrue(find_non_ascii_symbols(story_with_non_ascii))
        self.assertFalse(find_non_ascii_symbols(story_with_allowed_non_ascii))
        self.assertFalse(find_non_ascii_symbols(story_without_non_ascii))

    def test_clean_dataset(self):
        # Testing with min_length set to 6
        cleaned_stories = clean_dataset(self.raw_stories, min_length=self.min_length)
        print(f"Cleaned stories (min_length={self.min_length}): {cleaned_stories}")
        self.assertEqual(cleaned_stories, self.cleaned_stories_min_length_6)

        # Testing with a higher min_length
        cleaned_stories_high_min_length = clean_dataset(self.raw_stories, min_length=50)
        print(f"Cleaned stories (min_length=50): {cleaned_stories_high_min_length}")
        self.assertEqual(cleaned_stories_high_min_length, self.cleaned_stories_min_length_50)


if __name__ == '__main__':
    unittest.main()
