import pickle
import sys
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer
from torch import Tensor
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from model_1 import TransformerModel, device, num_special_tokens
from torch.utils.data import Dataset


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
    return load_dataset("roneneldan/TinyStories", split=split)[start:end]['text']


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


def get_vocabulary_idx(story_list: list[str], max_words: int | None = None) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocab_freq = get_vocabulary_frequencies(story_list)
    if max_words is not None:
        vocab = {}
        for _, (k, v) in zip(range(max_words - num_special_tokens),
                             sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)):
            vocab[k] = v
        vocab_freq = vocab

    vocab_freq['<unk>'] = 0  # Placeholder for tokens that do not appear in the story
    vocab_freq['<pad>'] = 0  # Pad token for batching
    return {k: idx for idx, k in enumerate(vocab_freq.keys())}


def map_story_to_tensor(story: str, vocab: dict, tokenizer) -> Tensor:
    """
    Maps a story to a Tensor of Integers, according to the index in the vocabulary
    """
    default = vocab["<unk>"]
    return torch.tensor([vocab.get(word, default) for word in tokenizer(story)], dtype=torch.int32)


def clean_stories(story_list: list[str]) -> list[str]:
    """
    Fixes certain encoding errors in the stories and removes all stories that either are empty or still contain
    non-ascii characters. Removes approximately 0.8% of all stories (802 of first 100K stories).
    """
    for idx, story in enumerate(story_list):
        if 'â' in story:
            story = remove_enc_errors(story)
            story_list[idx] = story
    return [story for story in story_list if story.isascii() and len(story) > 1]


def remove_enc_errors(story: str) -> str:
    story = story.replace('â€™', "'")
    story = story.replace("â€“", "-")
    story = story.replace("â€”", " - ")
    story = story.replace("â€š", ",")
    story = story.replace("â€œ", '"')
    story = story.replace('â€', '"')
    return story


def tokens_to_story(token_list: list[str]) -> str:
    if not nltk.download('punkt', quiet=True):
        nltk.download('punkt')
    sentence = TreebankWordDetokenizer().detokenize(token_list)
    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(sentence)

    story = " ".join([s.capitalize() for s in sentences])
    story = re.sub(r'\si\s', r' I ', story)  # Fix capitalization of 'i'
    story = re.sub(r"n' t", "n't", story)  # Fix contraction
    story = re.sub(r"' s\s", "'s ", story)  # Fix possessive
    story = re.sub(r"' d\s", "'d ", story)

    # List of all names in the vocabulary
    names = {'ben', 'bob', 'emily', 'joe', 'john', 'lily', 'lucy', 'max', 'mia', 'sam', 'sara', 'sarah', 'timmy', 'tom'}
    # ToDo: can be more efficient
    for name in names:
        story = story.replace(name, name.title())

    return story


def prompt_model(model_name: str, start_token: str, length: int = 50) -> str:
    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load(f'trained_models/{model_name}.pth').to(device)
    except FileNotFoundError:
        model = TransformerModel(len(vocab)).to(device)

    input_tensor = torch.tensor(vocab[start_token], dtype=torch.int32)
    input_tensor = input_tensor.view(1, -1)
    tl = model.generate_tokens(input_tensor.to(device), length)

    story_list = []
    for batch in tl:
        token_list = []
        for val in batch:
            token = vocab_rev[val.item()]
            token_list.append(token)
        story_list.append(tokens_to_story(token_list))
    return story_list[0]    # ToDo: maybe adjust function for generating multiple stories at once


class TinyStories(Dataset):
    def __init__(self, vocabulary, tokenizer=get_tokenizer('basic_english'), split: str = "train", max_seq_len=None):
        self.stories = load_dataset("roneneldan/TinyStories", num_proc=4, split=split)
        self.vocab = vocabulary
        self.tokenizer = tokenizer

        self.unk_token = self.vocab["<unk>"]
        self.pad_token = self.vocab["<pad>"]
        self.max_seq_len = max_seq_len

    def get_batch(self, sequences) -> tuple[Tensor, Tensor]:
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous()

    def __getitem__(self, index) -> Tensor:
        story = self.stories[index]['text']
        if 'â' in story:
            story = remove_enc_errors(story)

        if self.max_seq_len is None:
            token_list = [self.vocab.get(word, self.unk_token) for word in self.tokenizer(story)]
        else:
            token_list = []
            for _, word in zip(range(self.max_seq_len + 1), self.tokenizer(story)):
                token_list.append(self.vocab.get(word, self.unk_token))

        data = torch.tensor(token_list, dtype=torch.int64)
        if len(data) < 2:
            print(f"'useless' story (of length {len(data)}) at index {index}", file=sys.stderr)
        return data

    def __len__(self):
        return len(self.stories)


def save_vocabulary(vocab: dict[str, int], filename: str = "trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def load_vocabulary(filename: str = "trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    vocab = load_vocabulary()
    data = TinyStories(vocab)
    dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=2)
    print(len(dataloader))

    # stories = load_tiny_stories(100)
    # stories = clean_stories(stories)
    # print("Number of stories:", len(stories))
    #
    # token_dict = get_vocabulary_frequencies(stories)
    # token_dict_sorted = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    # print(f"Number of tokens: {len(token_dict)}")
    # print("Token frequency:", token_dict_sorted)
