from collections import Counter
import numpy as np
from io_utils import (
    load_from_disk,
    create_vocabulary,
    save_vocabulary,
    load_vocabulary,
    get_token_frequencies,
    SpacyTokenizer
)
import matplotlib.pyplot as plt


def calculate_token_coverage(stories, tokenizer):
    """
    Calculate the cumulative token coverage.

    :param stories: List of texts to analyze.
    :param tokenizer: Function to tokenize the stories.

    :return: cumulative_coverage (list): List of cumulative coverage percentages for tokens.
    """
    token_frequencies = get_token_frequencies(stories, tokenizer)

    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_frequencies.items(), key=lambda item: item[1], reverse=True)
    total_tokens = sum(token_frequencies.values())

    # Calculate cumulative coverage of tokens
    cumulative_coverage = []
    cumulative_count = 0
    for token, count in sorted_tokens:
        cumulative_count += count
        cumulative_coverage.append(cumulative_count / total_tokens)

    return cumulative_coverage


def calculate_statistics(stories, tokenizer):
    """
    Calculate various statistics for a given set of stories.

    :param stories: List of texts to analyze.
    :param tokenizer: Function to tokenize the stories.

    :return: total_tokens (int): Total number of tokens across all stories.
    :return: unique_tokens (int): Number of unique tokens.
    :return: avg_seq_length (float): Average number of tokens per story.
    :return: std_dev_seq_length (float): Standard deviation of sequence lengths.
    :return: token_counts (list): List of token counts per story.
    :return: lower_50_percent_max_frequency (int): Max frequency in the lower 50% of token frequencies.
    :return: lower_25_percent_max_frequency (int): Max frequency in the lower 25% of token frequencies.
    :return: token_coverage (list): Cumulative token coverage.
    """
    token_counts = [len(tokenizer(story)) for story in stories]
    total_tokens = sum(token_counts)
    all_tokens = [token for story in stories for token in tokenizer(story)]
    unique_tokens = len(set(all_tokens))
    avg_seq_length = total_tokens / len(stories)

    # Calculate standard deviation of sequence lengths
    std_dev_seq_length = np.std(token_counts, ddof=0)

    # Calculate frequency of each unique token
    token_frequencies = Counter(all_tokens)

    # Sort tokens by frequency
    sorted_token_frequencies = sorted(token_frequencies.values())

    # Calculate max frequency for the lower 50% of the tokens
    lower_50_percent_index = len(sorted_token_frequencies) // 2
    lower_50_percent_max_frequency = sorted_token_frequencies[lower_50_percent_index - 1]

    # Calculate max frequency for the lower 25% of the tokens
    lower_25_percent_index = len(sorted_token_frequencies) // 4
    lower_25_percent_max_frequency = sorted_token_frequencies[lower_25_percent_index - 1]

    # Calculate token coverage
    token_coverage = calculate_token_coverage(stories, tokenizer)

    return total_tokens, unique_tokens, avg_seq_length, std_dev_seq_length, token_counts, lower_50_percent_max_frequency, lower_25_percent_max_frequency, token_coverage


if __name__ == "__main__":
    # Load datasets
    dataset = load_from_disk("data/TinyStories")
    train_stories = dataset["train"][:]["text"]
    test_stories = dataset["test"][:]["text"]
    validation_stories = dataset["validation"][:]["text"]

    # Create and save vocabulary
    vocab = create_vocabulary(train_stories, SpacyTokenizer(), 2048)
    save_vocabulary(vocab)
    loaded_vocab = load_vocabulary("trained_models/vocabulary.pkl")
    print(f"Vocab with 2048 tokens: {loaded_vocab}")

    # Tokenizer
    tokenizer = SpacyTokenizer()

    # Calculate statistics for training set
    train_stats = calculate_statistics(train_stories, tokenizer)
    train_total_tokens, train_unique_tokens, train_avg_seq_length, train_std_dev_seq_length, train_token_counts, train_lower_50_percent_max_frequency, train_lower_25_percent_max_frequency, train_token_coverage = train_stats
    print("Training Set Statistics:")
    print(f"Number of tokens: {train_total_tokens}")
    print(f"Number of unique tokens: {train_unique_tokens}")
    print(f"Average sequence length: {train_avg_seq_length:.1f} (±{train_std_dev_seq_length:.1f})")
    print(f"Max frequency of the lower 50% of unique tokens: {train_lower_50_percent_max_frequency}")
    print(f"Max frequency of the lower 25% of unique tokens: {train_lower_25_percent_max_frequency}")

    # Calculate statistics for test set
    test_stats = calculate_statistics(test_stories, tokenizer)
    test_total_tokens, test_unique_tokens, test_avg_seq_length, test_std_dev_seq_length, test_token_counts, test_lower_50_percent_max_frequency, test_lower_25_percent_max_frequency, test_token_coverage = test_stats
    print("Test Set Statistics:")
    print(f"Number of tokens: {test_total_tokens}")
    print(f"Number of unique tokens: {test_unique_tokens}")
    print(f"Average sequence length: {test_avg_seq_length:.1f} (±{test_std_dev_seq_length:.1f})")
    print(f"Max frequency of the lower 50% of unique tokens: {test_lower_50_percent_max_frequency}")
    print(f"Max frequency of the lower 25% of unique tokens: {test_lower_25_percent_max_frequency}")

    # Calculate statistics for validation set
    validation_stats = calculate_statistics(validation_stories, tokenizer)
    validation_total_tokens, validation_unique_tokens, validation_avg_seq_length, validation_std_dev_seq_length, validation_token_counts, validation_lower_50_percent_max_frequency, validation_lower_25_percent_max_frequency, validation_token_coverage = validation_stats
    print("Validation Set Statistics:")
    print(f"Number of tokens: {validation_total_tokens}")
    print(f"Number of unique tokens: {validation_unique_tokens}")
    print(f"Average sequence length: {validation_avg_seq_length:.1f} (±{validation_std_dev_seq_length:.1f})")
    print(f"Max frequency of the lower 50% of unique tokens: {validation_lower_50_percent_max_frequency}")
    print(f"Max frequency of the lower 25% of unique tokens: {validation_lower_25_percent_max_frequency}")

    # Combine train, test, and validation sets to calculate statistics for the entire dataset
    combined_stories = train_stories + test_stories + validation_stories
    combined_stats = calculate_statistics(combined_stories, tokenizer)
    combined_total_tokens, combined_unique_tokens, combined_avg_seq_length, combined_std_dev_seq_length, combined_token_counts, combined_lower_50_percent_max_frequency, combined_lower_25_percent_max_frequency, combined_token_coverage = combined_stats
    print("Combined Dataset Statistics:")
    print(f"Number of tokens: {combined_total_tokens}")
    print(f"Number of unique tokens: {combined_unique_tokens}")
    print(f"Average sequence length: {combined_avg_seq_length:.1f} (±{combined_std_dev_seq_length:.1f})")
    print(f"Max frequency of the lower 50% of unique tokens: {combined_lower_50_percent_max_frequency}")
    print(f"Max frequency of the lower 25% of unique tokens: {combined_lower_25_percent_max_frequency}")

    # Verify dataset sizes
    print(f"Number of training stories: {len(train_stories)}")
    print(f"Number of test stories: {len(test_stories)}")
    print(f"Number of validation stories: {len(validation_stories)}")

    # Plot histogram with more bins for smoothness and enhanced visibility
    plt.figure(figsize=(10, 6), dpi=300)
    bins = range(0, 700, 1)
    plt.hist(train_token_counts, bins=bins, alpha=0.7, label='Training set', density=False, color='lightblue')
    plt.hist(test_token_counts, bins=bins, alpha=0.7, label='Test set', density=False, color='red')
    plt.hist(validation_token_counts, bins=bins, alpha=0.7, label='Validation set', density=False, color='green')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of story lengths')
    plt.legend(loc='upper right')
    plt.show()

    # Print the sum of frequencies to verify
    print("Sum of training set frequencies:", sum(train_token_counts))
    print("Sum of test set frequencies:", sum(test_token_counts))
    print("Sum of validation set frequencies:", sum(validation_token_counts))

    # Print the peak frequency value
    peak_frequency_training = max(train_token_counts)
    peak_bin_training = train_token_counts[np.argmax(train_token_counts)]
    print(f"Peak frequency in training set: {peak_frequency_training} in bin starting at {peak_bin_training} tokens")

    peak_frequency_test = max(test_token_counts)
    peak_bin_test = test_token_counts[np.argmax(test_token_counts)]
    print(f"Peak frequency in test set: {peak_frequency_test} in bin starting at {peak_bin_test} tokens")

    peak_frequency_validation = max(validation_token_counts)
    peak_bin_validation = validation_token_counts[np.argmax(validation_token_counts)]
    print(
        f"Peak frequency in validation set: {peak_frequency_validation} in bin starting at {peak_bin_validation} tokens")

    # Plot coverage graph with enhanced visibility
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_token_coverage, label='Training set', alpha=0.7, color='lightblue')
    plt.plot(test_token_coverage, label='Test set', alpha=0.7, color='red')
    plt.plot(validation_token_coverage, label='Validation set', alpha=0.7, color='green')
    plt.axvline(x=2048)
    plt.xscale('log')
    plt.xlabel('Used number of unique tokens (log-scale)')
    plt.ylabel('Percentage of coverage')
    plt.title('Coverage of tokens w.r.t. vocabulary size')
    plt.legend(loc='lower right')
    plt.show()
