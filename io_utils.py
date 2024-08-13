import sys
import os
import re
import pickle
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import spacy
from datasets import load_from_disk
from data.preprocess_dataset import clean_dataset
from model_1 import device, num_special_tokens, generate_tokens, generate_tokens_beam, generate_tokens_beam_multinomial


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


class SpacyTokenizer:
    def __init__(self):
        self.tokenizer = spacy.blank('en')

    def __call__(self, story: str):
        return [token.text for token in self.tokenizer(story)]


def get_token_frequencies(story_list: list[str], tokenizer=SpacyTokenizer(),
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


def create_vocabulary(story_list: list[str], tokenizer=SpacyTokenizer(),
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
    return torch.tensor([vocab.get(token, default) for token in tokenizer(story)], dtype=torch.int32)


def tokens_to_story(token_list: list[str]) -> str:
    story = " ".join(token_list)

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
                if story_list[i - 1] == ' ':
                    story_list[i - 1] = ''
            elif i != len(story_list) - 1:  # Opening quote
                if story_list[i + 1] == ' ':
                    story_list[i + 1] = ''
            in_quote = not in_quote

    story = ''.join(story_list)
    story = re.sub(r'(,"\s*)([A-Z])', lambda x: x.group(1) + x.group(2).lower(), story)

    # handle capitalization after closing quotes
    story = re.sub(r'"\s+([a-z])', lambda x: '" ' + x.group(1).lower(), story)
    story = re.sub(r'",\s+([a-z])', lambda x: '", ' + x.group(1).lower(), story)

    # names obtained from GPT-4o by providing list of vocabulary and asking for names:
    names = {'alice', 'amy', 'anna', 'ben', 'bella', 'benny', 'billy', 'bob', 'bobo', 'daisy', 'dave', 'emily',
             'emma', 'ellie', 'ella', 'george', 'jack', 'jake', 'jane', 'jen', 'jenny', 'jim', 'jimmy', 'joe',
             'john', 'johnny', 'leo', 'lila', 'lily', 'lisa', 'lola', 'lucy', 'mandy', 'mark', 'mary', 'max', 'mia',
             'mike', 'molly', 'pete', 'peter', 'polly', 'rex', 'sally', 'sam', 'sammy', 'sara', 'sarah', 'sophie',
             'susie', 'tim', 'timmy', 'tom', 'tommy', 'toby'}

    # replace names with capitalized names
    story = re.sub(r'\b(' + '|'.join(names) + r')\b', lambda x: x.group().capitalize(), story)

    story = re.sub(r'\bi\b', r' I ', story)  # Fix capitalization of 'i'
    
    return story


def prompt_model(model_name: str, start_str: str, length: int = 250, temperature: float = 1.0, method: str = "default",
                 beam_width: int = 5, top_k: int = 30) -> str:
    vocab = load_vocabulary()
    vocab_rev = {v: k for k, v in vocab.items()}
    tokenizer = SpacyTokenizer()
    model_path = get_absolute_path(f"trained_models/{model_name}.pth")

    try:
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    except FileNotFoundError:
        print(f"Model '{model_path}' could not be found", file=sys.stderr)
        sys.exit(1)

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    default = vocab["<unk>"]
    bos_token = vocab.get("<bos>")
    if bos_token is None:
        raise ValueError("<bos> token is not in vocabulary!")

    start_str = clean_dataset([start_str], True)[0].lower()
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

    if tl.size(0) != 1:
        print(f"Batch size must be 1, got {tl.size(0)}", file=sys.stderr)
    tl = tl[:, 1:]  # Strip <bos> token

    token_list = []
    for val in tl[0]:
        token = vocab_rev[val.item()]
        token_list.append(token)
    return tokens_to_story(token_list)


class TinyStories(Dataset):
    def __init__(self, vocabulary: dict, tokenizer=SpacyTokenizer(), split: str = "train",
                 dataset_path: str = "data/TinyStories", max_seq_len: int | None = None):
        self.stories = load_from_disk(dataset_path)[split]
        self.vocab = vocabulary
        self.tokenizer = tokenizer

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
        lengths = torch.tensor([s.shape[0] - 1 for s in sequences])
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous(), lengths

    def get_stories(self):
        return self.stories

    def __getitem__(self, index: int) -> Tensor:
        story = self.stories[index]['text']

        token_list = [self.bos_token]
        tokens = self.tokenizer(story.lower())
        for _, token in zip(range(self.max_seq_len), tokens):
            token_list.append(self.vocab.get(token, self.unk_token))

        if len(token_list) <= self.max_seq_len:
            token_list.append(self.eos_token)

        data = torch.tensor(token_list, dtype=torch.int64)
        return data

    def __len__(self):
        return len(self.stories)


def save_vocabulary(vocab: dict[str, int], filename: str = "trained_models/vocabulary.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    abs_path = os.path.abspath(os.path.join(script_dir, relative_path))
    return abs_path


def load_vocabulary(filename: str = "trained_models/vocabulary.pkl") -> dict[str, int]:
    abs_filename = get_absolute_path(filename)
    if not os.path.exists(abs_filename):
        raise FileNotFoundError(f"Vocabulary file not found: {abs_filename}")
    with open(abs_filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Load dataset
    dataset = load_from_disk("data/TinyStories")
    train_stories = dataset["train"][:]["text"]

    # Create and save vocabulary
    vocab = create_vocabulary(train_stories, max_words=2048)
    save_vocabulary(vocab)
    loaded_vocab = load_vocabulary("trained_models/vocabulary.pkl")
    print(f"Vocab with 2048 tokens: {loaded_vocab}")
