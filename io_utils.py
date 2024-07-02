import pickle
import sys
import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import torchtext
#torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from torch import Tensor
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from model_1 import device, num_special_tokens, generate_tokens, generate_tokens_beam, generate_tokens_beam_multinomial
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_tiny_stories(end: int, start: int = 0, split: str = "train") -> list[str]:
    """
    (Down-)Loads the TinyStories Dataset and returns the entries 'start' to 'end - 1' from the chosen split
    (i.e., returns 'end - start' many stories).
    :param end: The index of the story after the last story to be loaded from the dataset (story with index 'end' will
        not be returned)
    :param start: The index of the first story to be loaded from the dataset
    :param split: Choice between 'train' or 'validation' split
    :return: List of stories
    """
    return load_from_disk("data/TinyStories")[split][start:end]['text']


def load_from_file(filename: str) -> list[str]:
    """
    Loads the dataset from 'filename'.
    """
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read().split('\n<end>\n\n')[:-1]


def save_to_file(filename: str, story_list: list):
    """
    Saves a list of stories to the file 'filename'.
    """
    with open(filename, 'w', encoding="utf-8") as f:
        for item in story_list:
            item = "\n".join(item.split("\n\n"))
            f.write(f"{item}\n<end>\n\n")


def get_token_frequencies(story_list: list[str], tokenizer=get_tokenizer('spacy', language='en_core_web_sm'),
                          split_on_hyphen: bool = True) -> dict[str, int]:
    """
    Returns a dict of all tokens and their absolute frequencies
    """
    vocabulary = {}
    for story in story_list:
        tokens = tokenizer(story.lower())
        for token in tokens:
            token = token.strip("*")
            if split_on_hyphen is True and "-" in token:
                token_split = token.split("-")
                vocabulary.setdefault("-", 0)
                vocabulary["-"] += len(token_split) - 1
                for split_token in token_split:
                    vocabulary.setdefault(split_token, 0)
                    vocabulary[split_token] += 1
            else:
                vocabulary.setdefault(token, 0)
                vocabulary[token] += 1
    return vocabulary


def create_vocabulary(story_list: list[str], tokenizer=get_tokenizer('spacy', language='en_core_web_sm'),
                      max_words: int | None = None) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocab_freq = get_token_frequencies(story_list, tokenizer, split_on_hyphen=False)
    if max_words is not None:
        vocab = {}
        for _, (k, v) in zip(range(max_words - num_special_tokens),
                             sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)):
            vocab[k] = v
        vocab_freq = vocab

    vocab_freq["<eos>"] = 0  # end of sequence token
    vocab_freq['<unk>'] = 0  # Placeholder for tokens that do not appear in the story
    vocab_freq['<pad>'] = 0  # Pad token for batching
    return {k: idx for idx, k in enumerate(vocab_freq.keys())}


def map_story_to_tensor(story: str, vocab: dict, tokenizer) -> Tensor:
    """
    Maps a story to a Tensor of Integers, according to the index in the vocabulary
    """
    default = vocab["<unk>"]
    return torch.tensor([vocab.get(word, default) for word in tokenizer(story)], dtype=torch.int32)


# def clean_stories(story_list: list[str]) -> list[str]:
#     """
#     Fixes certain encoding errors in the stories and removes all stories that either are empty or still contain
#     non-ascii characters. Removes approximately 0.8% of all stories (802 of first 100K stories).
#     """
#     story_set = set(story_list)
#     story_set.discard("")
#     new_story_list = []
#
#     for idx, story in enumerate(story_set):
#         if 'â' in story:
#             story = remove_enc_errors(story)
#         if story.isascii() is True:
#             new_story_list.append(story)
#     return new_story_list


# def remove_enc_errors(story: str) -> str:
#     story = story.replace('â€™', "'")
#     story = story.replace("â€“", "-")
#     story = story.replace("â€”", " - ")
#     story = story.replace("â€š", ",")
#     story = story.replace("â€œ", '"')
#     story = story.replace('â€', '"')
#     return story


def tokens_to_story(token_list: list[str]) -> str:

    story = " ".join(token_list)
    
    story = re.sub(r'\si\s', r' I ', story)  # Fix capitalization of 'i'
    
    # Fix contraction, possessive, 'd, 're
    patterns = {
        r"n' t": "n't",
        r"(\w) n't": r"\1n't",
        r"' s\s": "'s ",
        r"(\w) 's": r"\1's",
        r"' d\s": "'d ",
        r"(\w) 'd": r"\1'd",
        r"' re\s": "'re ",
        r"(\w) 're": r"\1're",
        r"' m\s": "'m ",
        r"(\w) 'm": r"\1'm",
        r"' ve\s": "'ve ",
        r"(\w) 've": r"\1've",
        r"' ll\s": "'ll ",
        r"(\w) 'll": r"\1'll"
    }

    for pattern, replacement in patterns.items():
        story = re.sub(pattern, replacement, story)
    
    # Fix spaces around punctuation
    story = re.sub(r'\s([?.!,;:](?:\s|$))', r'\1', story)

    # unify quotation marks
    story = re.sub(r'“|”', '"', story)

    # capitalize first letter of each sentence
    story = story[0].upper() + story[1:]

    story = re.sub(r'([.!?"]\s*)([a-z])', lambda x: x.group(1) + x.group(2).upper(), story)

    # handle space before and after " based on appearance (cannot handle nested quotes)
    in_quote = False
    # loop through all characters in the story and delete unnecessary spaces
    # if closing quote: delete space before quote
    # if opening quote: delete space after quote
    story_list = list(story)
    for i, char in enumerate(story_list):
        if char == '"':
            if in_quote:  # Closing quote
                if story_list[i-1] == ' ':
                    story_list[i-1] = ''
            elif i != len(story_list) - 1:  # Opening quote
                if story_list[i+1] == ' ':
                    story_list[i+1] = ''
            in_quote = not in_quote
    
    story = ''.join(story_list)

    story = re.sub(r'(,"\s*)([A-Z])', lambda x: x.group(1) + x.group(2).lower(), story)

    names = {'ben', 'billy', 'bob', 'emily', 'jack', 'joe', 'john', 'lily', 'lucy', 'max', 'mia', 'polly', 'sam', 'sara', 'sarah', 'timmy', 'tom'}
    # names obtained from GPT-4o by providing list of vocabulary and asking for names:
    names_gpt = ['alice', 'amy', 'anna', 'ben', 'bella', 'benny', 'billy', 'bob', 'bobo', 'daisy', 'dave', 'emma', 'ellie', 'ella', 'george', 'jack', 'jake', 'jane', 'jen', 'jenny', 'jim', 'jimmy', 'joe', 'john', 'johnny', 'leo', 'lila', 'lily', 'lisa', 'lola', 'lucy', 'mandy', 'mark', 'mary', 'max', 'mia', 'mike', 'molly', 'pete', 'peter', 'rex', 'sally', 'sam', 'sammy', 'sara', 'sarah', 'sophie', 'susie', 'tim', 'timmy', 'tom', 'tommy', 'toby']
    names = names.union(names_gpt)

    # replace names with capitalized names
    story = re.sub(r'\b(' + '|'.join(names) + r')\b', lambda x: x.group().capitalize(), story)

    return story


def prompt_model(model_name: str, start_str: str, length: int = 250, temperature: float = 1.0, method: str = "default", beam_width: int = 5, top_k: int = 50, sampling_after: int = 5) -> str:
    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}

    try:
        model = torch.load(f'trained_models/{model_name}.pth').to(device)
    except FileNotFoundError:
        print(f"Model 'trained_models/{model_name}.pth could not be found", file=sys.stderr)
        sys.exit(1)

        # Ensure nhead attribute is present
    #if not hasattr(model, 'nhead'):
        #raise AttributeError(f"Loaded model does not have 'nhead' attribute (cannot use ROPE).")

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    default = vocab["<unk>"]

    input_tensor = torch.tensor([vocab.get(token, default) for token in tokenizer(start_str.lower())],
                                dtype=torch.int32)
    input_tensor = input_tensor.view(1, -1)

    match method:
        case "beam":
            tl = generate_tokens_beam(model, input_tensor, beam_width, length, eos_token=vocab.get("<eos>"),
                                      temperature=temperature)
        case "beam_multinomial":
            tl = generate_tokens_beam_multinomial(model, input_tensor, beam_width, length, eos_token=vocab.get("<eos>"),
                                                  temperature=temperature, top_k=top_k)
        case _:
            tl = generate_tokens(model, input_tensor.to(device), length, eos_token=vocab.get("<eos>"),
                                 temperature=temperature)

    story_list = []
    for batch in tl:
        token_list = []
        for val in batch:
            token = vocab_rev[val.item()]
            token_list.append(token)
        story_list.append(tokens_to_story(token_list))
    return story_list[0]  # ToDo: maybe adjust function for generating multiple stories at once



class TinyStories(Dataset):
    def __init__(self, vocabulary: dict, tokenizer=get_tokenizer('spacy', language='en_core_web_sm'),
                 split: str = "train", max_seq_len: int | None = None, split_on_hyphen: bool = False):
        self.stories = load_from_disk("data/TinyStories")[split]
        self.vocab = vocabulary
        self.tokenizer = tokenizer
        self.split_on_hyphen = split_on_hyphen

        self.unk_token = self.vocab["<unk>"]
        self.pad_token = self.vocab["<pad>"]
        self.max_seq_len = max_seq_len if max_seq_len is not None else 10000

    def get_batch(self, sequences: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        lengths = torch.cat([torch.tensor(s.shape)-1 for s in sequences])
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous(), lengths

    def __getitem__(self, index: int) -> Tensor:
        story = self.stories[index]['text']

        token_list = []
        tokens = self.tokenizer(story.lower())
        for _, word in zip(range(self.max_seq_len + 1), tokens):
            token_list.append(self.vocab.get(word, self.unk_token))

        if len(token_list) <= self.max_seq_len:
            eos_token = self.vocab.get("<eos>")
            if eos_token is None:
                raise ValueError("<eos> token is not found in the vocabulary.")
            token_list.append(eos_token)
        # token_list = token_list[:self.max_seq_len + 1]

        data = torch.tensor(token_list, dtype=torch.int64)
        return data

    def __len__(self):
        return len(self.stories)


def save_vocabulary(vocab: dict[str, int], filename: str = "trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def load_vocabulary(filename: str = "trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as file:
        return pickle.load(file)


def calculate_token_coverage(stories, tokenizer):
    token_frequencies = get_token_frequencies(stories, tokenizer)

    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_frequencies.items(), key=lambda item: item[1], reverse=True)
    total_tokens = sum(token_frequencies.values())

    # Calculate cumulative coverage
    cumulative_coverage = []
    cumulative_count = 0
    for token, count in sorted_tokens:
        cumulative_count += count
        cumulative_coverage.append(cumulative_count / total_tokens)

    return cumulative_coverage

def calculate_statistics(stories, tokenizer):
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

    # Create and save vocabulary
    vocab = create_vocabulary(train_stories, get_tokenizer('spacy', language='en_core_web_sm'), 2048)
    save_vocabulary(vocab)
    loaded_vocab = load_vocabulary("trained_models/vocabulary.pkl")
    print(f"Vocab with 2048 tokens: {loaded_vocab}")

    # Tokenizer
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

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

    # Combine train and test sets to calculate statistics for the entire dataset (without validation)
    combined_stories = train_stories + test_stories
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

    # Plot histogram with more bins for smoothness
    plt.figure(figsize=(10, 6))
    bins = range(0, 700, 1)
    train_hist = plt.hist(train_token_counts, bins=bins, alpha=0.7, label='Training set', density=False)
    test_hist = plt.hist(test_token_counts, bins=bins, alpha=0.7, label='Test set', density=False)
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of story lengths')
    plt.legend(loc='upper right')
    plt.show()

    # Print the sum of frequencies to verify
    print("Sum of training set frequencies:", sum(train_hist[0]))
    print("Sum of test set frequencies:", sum(test_hist[0]))

    # Print the peak frequency value
    peak_frequency_training = max(train_hist[0])
    peak_bin_training = train_hist[1][np.argmax(train_hist[0])]
    print(f"Peak frequency in training set: {peak_frequency_training} in bin starting at {peak_bin_training} tokens")

    peak_frequency_test = max(test_hist[0])
    peak_bin_test = test_hist[1][np.argmax(test_hist[0])]
    print(f"Peak frequency in test set: {peak_frequency_test} in bin starting at {peak_bin_test} tokens")

    # Plot coverage graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_token_coverage, label='Training set', alpha=0.7)
    plt.plot(test_token_coverage, label='Test set', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Used number of unique tokens (log-scale)')
    plt.ylabel('Percentage of coverage')
    plt.title('Coverage of tokens w.r.t. vocabulary size')
    plt.legend(loc='lower right')
    plt.show()
