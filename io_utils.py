import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torch import Tensor
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

from model_1 import TransformerModel


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


def get_vocabulary_idx(story_list: list, max_words: int | None = None, eos: bool = False) -> dict[str, int]:
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

    if eos:
        vocab_freq['<eos>'] = len(vocab_freq)
    vocab_freq['<pad>'] = len(vocab_freq)
    return {k: idx for idx, k in enumerate(vocab_freq.keys())}


def map_story_to_tensor(story: str, vocab: dict, tokenizer) -> Tensor:
    """
    Maps a story to a Tensor of Integers, according to the index in the vocabulary
    """
    default = vocab["<unk>"]
    return torch.tensor([vocab.get(word, default) for word in tokenizer(story)], dtype=torch.int64)


def map_stories_to_tensor(stories, vocab, tokenizer, max_length=None) -> Tensor:
    """
    Converts a list of stories to a batched tensor.

    Args:
        stories (list of str): The list of story strings to convert.
        vocab (dict): A dictionary mapping words to indices.
        tokenizer (callable): Function to tokenize each story.
        max_length (int, optional): The maximum length of the stories in the batch. If not provided, use the length of the longest story.

    Returns:
        torch.Tensor: A batched tensor of shape (batch_size, max_length).
    """
    tensor_list = [map_story_to_tensor(story, vocab, tokenizer) for story in stories]

    if max_length is not None:
        tensors = [tensor[:max_length] for tensor in tensor_list]  # Truncate to max_length if specified
    else:
        max_length = max(tensor.size(0) for tensor in tensor_list)  # Find the longest tensor

    #tensor = pad_sequence(tensor_list, batch_first=True, padding_value=vocab['<pad>'])
    return tensor_list

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
    if not nltk.download('punkt', quiet=True):
        nltk.download('punkt')
    sentence = TreebankWordDetokenizer().detokenize(token_list)
    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(sentence)

    story = " ".join([s.capitalize() for s in sentences])
    story = re.sub(r'\si\s', r' I ', story) # Fix capitalization of 'i'
    story = re.sub(r"n' t", "n't", story) # Fix contraction
    story = re.sub(r"' s", "'s", story) # Fix possessive
    return story

def prompt_model(model, start_token: str, length: int = 50, end_on_eos: bool = False) -> str:
    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    if end_on_eos:
        eos_idx = vocab['<eos>']
    else:
        eos_idx = None

    try:
        model = torch.load(f'trained_models/{model}.pth')
    except FileNotFoundError:
        model = TransformerModel(len(vocab))
    
    tl = model.generate_tokens(torch.tensor(vocab[start_token], dtype=torch.int64), length, eos_idx)
    token_list = []
    for val in tl:
        token_list.append(vocab_rev[val.item()])
    return tokens_to_story(token_list)


def save_vocabulary(vocab: dict[str, int], filename="trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocabulary(filename="trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_vocabulary(vocab: dict[str, int], filename="trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocabulary(filename="trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    stories = load_tiny_stories(2)
    #stories = load_from_file("data/100stories.txt")
    stories = clean_stories(stories)
    # save_to_file("data/100stories.txt", stories)
    print(type(stories))
    print(stories)
    print("Number of stories:", len(stories))
    print(f"Story 1 size: {len(stories[0])}")
    print(f"Story 2 size: {len(stories[1])}\n")
    #print(f"Story 3 size: {len(stories[2])}\n")


    tokenizer = get_tokenizer('basic_english')

    token_dict = get_vocabulary_frequencies(stories)
    token_dict_sorted = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    vocab_index = get_vocabulary_idx(stories, 200)
    story_tensor = map_story_to_tensor(stories[1], vocab_index, tokenizer)

    i = 0
    for s in stories:
        print(f"Story {i}: {stories[i]}\n")
        i+=1
    print(f"Vocab Frequencies: {token_dict}")
    print(f"Vocab Frequencies sorted: {token_dict_sorted}")
    print(f"Number of tokens: {len(token_dict)}")
    print("Token frequency:", token_dict_sorted)
    print(f"Vocab Indices: {vocab_index}")
    print(f"Story Tensor: {story_tensor}")

    stories_tensor = map_stories_to_tensor(stories, vocab_index, tokenizer)
    print(f"Stories Tensor: {stories_tensor}")
    #print(stories_tensor.size())

