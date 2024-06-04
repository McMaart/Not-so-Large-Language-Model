import pickle
import sys
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torchtext
#torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from torch import Tensor
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from model_1 import device, num_special_tokens
from torch.utils.data import Dataset

num_val_stories = 75000


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


def get_token_frequencies(story_list: list[str], split_on_hyphen: bool = True) -> dict[str, int]:
    """
    Returns a dict of all tokens and their absolute frequencies
    """
    vocabulary = {}
    tokenizer = get_tokenizer('basic_english')
    for story in story_list:
        tokens = tokenizer(story)
        for token in tokens:
            token = token.strip("*").strip("_")
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


def create_vocabulary(story_list: list[str], max_words: int | None = None) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocab_freq = get_token_frequencies(story_list)
    if max_words is not None:
        vocab = {}
        for _, (k, v) in zip(range(max_words - num_special_tokens),
                             sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)):
            vocab[k] = v
        vocab_freq = vocab

    vocab_freq["<eos>"] = 0
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
    story_set = set(story_list)
    story_set.discard("")
    new_story_list = []

    for idx, story in enumerate(story_set):
        if 'â' in story:
            story = remove_enc_errors(story)
        if story.isascii() is True:
            new_story_list.append(story)
    return new_story_list


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
    # ToDo: can be more efficient ToDo: use regex instead (or, even better, directly capitalize the tokens),
    #  for fixing spelling mistakes such as botTom
    for name in names:
        story = story.replace(name, name.title())

    return story


def prompt_model(model_name: str, start_str: str, length: int = 250) -> str:
    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}

    try:
        model = torch.load(f'trained_models/{model_name}.pth').to(device)
    except FileNotFoundError:
        print(f"Model 'trained_models/{model_name}.pth could not be found", file=sys.stderr)
        sys.exit(1)

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    tokenizer = get_tokenizer("basic_english")
    default = vocab["<unk>"]

    input_tensor = torch.tensor([vocab.get(token, default) for token in tokenizer(start_str)], dtype=torch.int32)
    input_tensor = input_tensor.view(1, -1)
    tl = model.generate_tokens(input_tensor.to(device), length, eos_token=vocab.get("<eos>"))

    story_list = []
    for batch in tl:
        token_list = []
        for val in batch:
            token = vocab_rev[val.item()]
            token_list.append(token)
        story_list.append(tokens_to_story(token_list))
    return story_list[0]  # ToDo: maybe adjust function for generating multiple stories at once


class TinyStories(Dataset):
    def __init__(self, vocabulary: dict, tokenizer=get_tokenizer('basic_english'), split: str = "train",
                 max_seq_len: int | None = None, split_on_hyphen: bool = True):
        self.stories = load_dataset("roneneldan/TinyStories", num_proc=4, split=split)
        self.vocab = vocabulary
        self.tokenizer = tokenizer
        self.split_on_hyphen = split_on_hyphen

        self.unk_token = self.vocab["<unk>"]
        self.pad_token = self.vocab["<pad>"]
        self.max_seq_len = max_seq_len if max_seq_len is not None else 10000

    def get_batch(self, sequences: list[Tensor]) -> tuple[Tensor, Tensor]:
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous()

    def __getitem__(self, index: int) -> Tensor:
        story = self.stories[index]['text']
        if 'â' in story:
            story = remove_enc_errors(story)

        token_list = []
        tokens = self.tokenizer(story)
        for _, word in zip(range(self.max_seq_len + 1), tokens):
            word = word.strip("*").strip("_")
            if self.split_on_hyphen is True and "-" in word:
                token_split = word.split("-")
                hyphen_token = self.vocab.get("-", self.unk_token)
                for split_token in token_split:
                    token_list.append(self.vocab.get(split_token, self.unk_token))
                    token_list.append(hyphen_token)
                token_list = token_list[:-1]
            else:
                token_list.append(self.vocab.get(word, self.unk_token))
        if len(token_list) <= self.max_seq_len:
            token_list.append(self.vocab.get("<eos>"))
        token_list = token_list[:self.max_seq_len + 1]

        data = torch.tensor(token_list, dtype=torch.int64)
        if len(data) < 2:
            print(f"'useless' story (of length {len(data)}) at index {index}", file=sys.stderr)
        return data

    def __len__(self):
        return len(self.stories) - num_val_stories


def save_vocabulary(vocab: dict[str, int], filename: str = "trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def load_vocabulary(filename: str = "trained_models/vocabulary.pkl") -> dict:
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    vocab = load_vocabulary()

    # data = TinyStories(load_vocabulary())
    # dataloader = DataLoader(data, batch_size=32, collate_fn=data.get_batch, num_workers=4,
    #                             pin_memory=True)
    # stories = clean_stories(stories)
    # print("Number of stories:", len(stories))
    #
    # token_dict = get_vocabulary_frequencies(stories)
    # token_dict_sorted = {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)}
    # print(f"Number of tokens: {len(token_dict)}")
    # print("Token frequency:", token_dict_sorted)
