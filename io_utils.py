import torch
from datasets import load_dataset
from torchtext.data import get_tokenizer
import re
from torch import Tensor


def load_tiny_stories(end: int, start: int = 0, split="train"):
    """
    (Down-)Loads the TinyStories Dataset and returns the entries 'start' to 'end - 1' from the chosen split
    (i.e., returns 'end - start' many stories)
    """
    return load_dataset("roneneldan/TinyStories")[split][start:end]['text']


def load_from_file(filename: str) -> list:
    """
    Loads a dataset from 'filename'
    """
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read().split('\n<end>\n\n')[:-1]


def save_to_file(filename: str, story_list: list):
    """
    Saves a list of stories to the file 'filename'
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


# ToDo: Assert that torchtext.data.utils.get_tokenizer works well, then remove the following manual implementation
def get_vocabulary_frequencies_2(story_list: list[str]) -> dict[str, int]:
    """
    Returns a dict of all tokens and their absolute frequencies
    Manual Implementation.
    """
    vocabulary = {}
    for item in story_list:
        tokens = re.split(r'\b(?![ \n])', item.lower())[1:]
        for token in tokens:
            token = token.strip('"').strip("'").strip("-").strip("â").strip()
            vocabulary.setdefault(token, 0)
            vocabulary[token] += 1
    return vocabulary


def get_vocabulary_idx(story_list: list) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocabulary_freq = get_vocabulary_frequencies(story_list)
    return {k: idx for idx, k in enumerate(vocabulary_freq.keys())}


def map_story_to_tensor(story: str, vocab: dict, tokenizer) -> Tensor:
    """
    Maps a story to a Tensor of Integers, according to the index in the vocabulary
    """
    return Tensor([vocab[word] for word in tokenizer(story)]).to(torch.int)


def tokens_to_story(token_list: list[str]) -> str:
    # ToDo: Remove whitespace after punctuation
    return " ".join(token_list)


if __name__ == "__main__":
    # stories = load_tiny_stories(100)
    # save_to_file("data/100stories.txt", stories)
    stories = load_from_file("data/100stories.txt")
    token_dict = get_vocabulary_frequencies(stories)
    # print("Stories:", stories)
    print(f"Number of tokens: {len(token_dict)}")
    print("Token frequency:", {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)})
