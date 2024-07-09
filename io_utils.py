import pickle
import sys
import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import torchtext
from data.preprocess_dataset import clean_dataset
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from torch import Tensor
# import nltk
# from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from model_1 import device, num_special_tokens, generate_tokens, generate_tokens_beam, generate_tokens_beam_multinomial
from torch.utils.data import Dataset
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
# import seaborn as sns


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

    # Note: <eos> must have a lower index in the vocabulary than the other tokens
    vocab_freq["<eos>"] = 0  # end of sequence token
    vocab_freq['<bos>'] = 0  # begin of sequence token
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


def prompt_model(model_name: str, start_str: str, length: int = 250, temperature: float = 1.0, method: str = "default", beam_width: int = 5, top_k: int = 30, sampling_after: int = 5) -> str:
    vocab = load_vocabulary()
    vocab_rev = {k: v for v, k in vocab.items()}

    model_path = get_absolute_path(f"../trained_models/{model_name}.pth")

    try:
        model = torch.load(f'../trained_models/{model_name}.pth', map_location=device).to(device)
    except FileNotFoundError:
        print(f"Model 'trained_models/{model_name}.pth could not be found", file=sys.stderr)
        sys.exit(1)

        # Ensure nhead attribute is present
    #if not hasattr(model, 'nhead'):
        #raise AttributeError(f"Loaded model does not have 'nhead' attribute (cannot use ROPE).")

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    default = vocab["<unk>"]
    bos_token = vocab.get("<bos>")
    if bos_token is None:
        raise ValueError("<bos> token is not in vocabulary!")

    start_str = clean_dataset([start_str], True)[0].lower()
    # todo: improve efficiency
    input_tensor = torch.tensor([bos_token] + [vocab.get(token, default) for token in tokenizer(start_str)],
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

    tl = tl[:, 1:]  # Strip <bos> token
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
        self.eos_token = self.vocab.get("<eos>")
        self.bos_token = self.vocab.get("<bos>")
        self.max_seq_len = max_seq_len if max_seq_len is not None else 10000

        if self.eos_token is None:
            raise ValueError("<eos> token is not found in the vocabulary.")
        elif self.bos_token is None:
            raise ValueError("<bos> token is not found in the vocabulary.")

    def get_batch(self, sequences: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        lengths = torch.cat([torch.tensor(s.shape)-1 for s in sequences])
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous(), lengths

    def __getitem__(self, index: int) -> Tensor:
        story = self.stories[index]['text']

        token_list = [self.bos_token]
        tokens = self.tokenizer(story.lower())
        for _, word in zip(range(self.max_seq_len), tokens):
            token_list.append(self.vocab.get(word, self.unk_token))

        if len(token_list) <= self.max_seq_len:
            token_list.append(self.eos_token)
        # token_list = token_list[:self.max_seq_len + 1]

        data = torch.tensor(token_list, dtype=torch.int64)
        return data

    def __len__(self):
        return len(self.stories)


def save_vocabulary(vocab: dict[str, int], filename: str = "trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)

def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)

def load_vocabulary(filename: str = "../trained_models/vocabulary.pkl") -> dict:
    if filename is None:
        filename = get_absolute_path("../trained_models/vocabulary.pkl")
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Load dataset
    dataset = load_from_disk("data/TinyStories")
    train_stories = dataset["train"][:]["text"]

    # Create and save vocabulary
    vocab = create_vocabulary(train_stories, get_tokenizer('spacy', language='en_core_web_sm'), 2048)
    save_vocabulary(vocab)
    loaded_vocab = load_vocabulary("trained_models/vocabulary.pkl")
    print(f"Vocab with 2048 tokens: {loaded_vocab}")

    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
