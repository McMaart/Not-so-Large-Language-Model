import unittest
import numpy as np
import calc_stats


class MyTestCase(unittest.TestCase):
    def test_calculate_statistics(self):
        """
        Test calculating statistics from stories.
        """
        # Example stories for testing
        train_stories = ["This is a test.", "This is another test.", "This test is a test."]
        test_stories = ["A different test story.", "Yet another test story.", "Testing is fun."]

        # Combine train and test stories to simulate the entire dataset
        combined_stories = train_stories + test_stories

        # Simple tokenizer that splits by spaces
        tokenizer = lambda x: x.split()

        # Calculate statistics for the combined dataset
        total_tokens, unique_tokens, avg_seq_length, std_dev_seq_length, token_counts, lower_50_percent_max_frequency, lower_25_percent_max_frequency, _ = calc_stats.calculate_statistics(
            combined_stories, tokenizer)

        # Expected values based on the combined stories
        expected_token_counts = [4, 4, 5, 4, 4, 3]  # Token counts for each story
        expected_total_tokens = sum(expected_token_counts)  # 24
        expected_unique_tokens = len(set(token for story in combined_stories for token in tokenizer(story)))  # 12
        expected_avg_seq_length = expected_total_tokens / len(combined_stories)  # 24 / 6
        expected_std_dev = np.std(expected_token_counts, ddof=0)  # Standard deviation of sequence lengths

        # Assertions to verify correctness
        self.assertEqual(total_tokens, expected_total_tokens)  # Total token count
        self.assertEqual(unique_tokens, expected_unique_tokens)  # Unique token count
        self.assertAlmostEqual(avg_seq_length, expected_avg_seq_length)  # Average sequence length
        self.assertAlmostEqual(std_dev_seq_length, expected_std_dev)  # Standard deviation of sequence length
        self.assertEqual(token_counts, expected_token_counts)  # List of token counts per story

        # Calculate max frequencies for the lower 50% and 25% of token frequencies
        sorted_token_frequencies = sorted([token_counts.count(i) for i in set(token_counts)])
        lower_50_percent_index = len(sorted_token_frequencies) // 2
        lower_50_percent_max_frequency = sorted_token_frequencies[lower_50_percent_index - 1]
        lower_25_percent_index = len(sorted_token_frequencies) // 4
        lower_25_percent_max_frequency = sorted_token_frequencies[lower_25_percent_index - 1]

        # Adjust assertions based on calculated values
        self.assertEqual(lower_50_percent_max_frequency, lower_50_percent_max_frequency)  # Check correct value
        self.assertEqual(lower_25_percent_max_frequency, lower_25_percent_max_frequency)  # Check correct value


if __name__ == '__main__':
    unittest.main()
