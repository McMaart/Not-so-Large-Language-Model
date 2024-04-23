import pickle
import torch
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torch import Tensor


def load_tiny_stories(end: int, start: int = 0, split="train"):
    """
    (Down-)Loads the TinyStories Dataset and returns the entries 'start' to 'end - 1' from the chosen split
    (i.e., returns 'end - start' many stories).
    """
    return load_dataset("roneneldan/TinyStories")[split][start:end]['text']


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


def get_vocabulary_frequencies(story_list: list[str]) -> dict[str, int]:
    """
    Returns a dict of all tokens and their absolute frequencies
    """
    vocabulary = {}
    tokenizer = get_tokenizer('basic_english')
    for story in story_list:
        tokens = tokenizer(story)
        for token in tokens:
            vocabulary.setdefault(token, 0)
            vocabulary[token] += 1
    return vocabulary


def get_vocabulary_idx(story_list: list, max_words: int | None = None) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocab_freq = get_vocabulary_frequencies(story_list)
    if max_words is not None:
        vocab = {}
        for _, (k, v) in zip(range(max_words-1), sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)):
            vocab[k] = v
        vocab_freq = vocab

    vocab_freq['<unk>'] = 0
    return {k: idx for idx, k in enumerate(vocab_freq.keys())}


def map_story_to_tensor(story: str, vocab: dict, tokenizer) -> Tensor:
    """
    Maps a story to a Tensor of Integers, according to the index in the vocabulary
    """
    default = vocab["<unk>"]
    return torch.tensor([vocab.get(word, default) for word in tokenizer(story)], dtype=torch.int64)


def clean_stories(story_list: list[str]) -> list[str]:
    """
    Fixes certain encoding errors in the stories and removes all stories that continue to have non-ascii characters.
    Removes approximately 0.8% of all stories (802 of first 100K stories).
    """
    for idx, story in enumerate(story_list):
        if 'â' in story:
            story = story.replace('â€œ', '"')
            story = story.replace('â€™', "'")
            story = story.replace('â€', '"')
            story_list[idx] = story
    return [story for story in story_list if story.isascii()]


def tokens_to_story(token_list: list[str]) -> str:
    # ToDo: Remove whitespace after punctuation
    return " ".join(token_list)


def save_vocabulary(vocab: dict[str, int], filename="trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocabulary(filename="trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # stories = load_tiny_stories(100)
    stories = load_from_file("data/100stories.txt")
    stories = clean_stories(stories)
    # save_to_file("data/100stories.txt", stories)
    print("Number of stories:", len(stories))

    token_dict = get_vocabulary_frequencies(stories)
    token_dict_sorted = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    print(f"Number of tokens: {len(token_dict)}")
    print("Token frequency:", token_dict_sorted)
