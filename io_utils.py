import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torch import Tensor
from torch.utils.data import Dataset
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import sys

from model_1 import TransformerModel, device
from model_2 import num_spec_tokens


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

def get_vocabulary_frequencies_2(story_list):
    vocabulary = {}
    tokenizer = get_tokenizer('basic_english')
    for story in story_list:
        tokens = tokenizer(story)
        for token in tokens:
            token = token.strip("*").strip("_")
            if "-" in token:
                token_split = token.split("-")
                vocabulary.setdefault("-", 0)
                for split_token in token_split:
                    vocabulary.setdefault(split_token, 0)
                    vocabulary[split_token] += 1
                    vocabulary["-"] += 1
            else:
                vocabulary.setdefault(token, 0)
                vocabulary[token] += 1
    return vocabulary

def get_vocabulary_idx(story_list: list[str], max_words: int | None = None) -> dict[str, int]:
    """
    Assigns an index to each word that appears in the list of stories
    """
    vocab_freq = get_vocabulary_frequencies_2(story_list)
    if max_words is not None:
        vocab = {}
        for _, (k, v) in zip(range(max_words - num_spec_tokens),
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

def prompt_model(model, start_token: str, length: int = 50, end_on_eos: bool = False, story_str=None) -> str:
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
    
    # tl = model.generate_tokens(torch.tensor(vocab[start_token], dtype=torch.int64), length, eos_idx)
    # # print(f"Generated tokens: {tl}")  # Debug print statement

    # token_list = []
    # for val in tl:
    #     token_list.append(vocab_rev[val.item()])
    # return tokens_to_story(token_list)

    if story_str:
        tokenizer = get_tokenizer("basic_english")
        default = vocab["<unk>"]
        input_tensor = torch.tensor([vocab.get(token, default) for token in tokenizer(story_str)], dtype=torch.int32)
        print(input_tensor.shape)
    else:
        input_tensor = torch.tensor(vocab[start_token], dtype=torch.int32)
    # print(f"Input tensor shape: {input_tensor.shape}")  # Debug print statement
    input_tensor = input_tensor.view(1, -1)
    tl = model.generate_tokens(input_tensor.to(device), length, eos_idx=vocab.get("<eos>"))
    # print(f"Generated tokens: {tl}")

    story_list = []
    token_list = []
    for batch in tl:
        # token_list = []
        for val in batch:
            token = vocab_rev[val.item()]
            token_list.append(token)
    story_list.append(tokens_to_story(token_list))
        # print("Tokenlist",token_list)
    return story_list[0]   

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

