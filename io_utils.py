import pickle
import torch
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torch import Tensor
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from model_1 import TransformerModel, device



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
    Assigns an index to each word that appears in the list of stories, including special tokens.
    """
    vocab_freq = get_vocabulary_frequencies(story_list)
    sorted_vocab = sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)

    # If a max_words limit is set, truncate the vocabulary
    if max_words is not None:
        sorted_vocab = sorted_vocab[:max_words]

    # Add special tokens
    vocab = {'<unk>': 0, '<pad>': 1}
    if eos:
        vocab['<eos>'] = 2
        start_idx = 3
    else:
        start_idx = 2

    # Add words to the vocab
    vocab.update({word: i + start_idx for i, (word, freq) in enumerate(sorted_vocab)})

    return vocab


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

def prompt_model(model_name: str, start_token: str, length: int = 50) -> str:
    vocab = load_vocabulary()
    names = {"bob", "lilly", "sarah", "tom", "lucy"}
    vocab_rev = {k: v for v, k in vocab.items()}
    try:
        model: TransformerModel = torch.load(f'trained_models/{model_name}.pth').to(device)
    except FileNotFoundError:
        model = TransformerModel(len(vocab)).to(device)

    input_tensor = torch.tensor(vocab[start_token], dtype=torch.int64).unsqueeze(0)
    tl = model.generate_tokens(input_tensor.to(device), length)

    token_list = []
    for val in tl:
        token = vocab_rev[val.item()]
        if token in names:
            token = token.title()
        token_list.append(token)
    return tokens_to_story(token_list)


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


