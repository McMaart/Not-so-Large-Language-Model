"""
Functions for creating a vocabulary, tokenization and postprocessing (ref. Section 3.1 and Section 3.5 in our report).
"""
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
from model_1 import device, num_special_tokens
from generate_stories import generate_tokens, generate_tokens_beam, generate_tokens_beam_multinomial


class SpacyTokenizer:
    """
    Wrapper around the SpaCy blank english tokenizer.
    """
    def __init__(self):
        self.tokenizer = spacy.blank('en')

    def __call__(self, story: str) -> list[str]:
        """
        Tokenizes a story into a list of word-level tokens.
        :param story: The story to be tokenized.
        :return: The list of tokens.
        """
        return [token.text for token in self.tokenizer(story)]


def get_token_frequencies(story_list: list[str], tokenizer=SpacyTokenizer()) -> dict[str, int]:
    """
    Returns a dictionary which contains all tokens appearing in story_list, and their absolute frequencies.
    :param story_list: The list of stores (e.g., the training dataset).
    :param tokenizer: The tokenizer for tokenizing the story.
    :return: The dictionary with string-tokens as keys and their frequencies as values.
    """
    frequencies = {}
    for story in story_list:
        tokens = tokenizer(story.lower())
        for token in tokens:
            frequencies.setdefault(token, 0)
            frequencies[token] += 1
    return frequencies


def create_vocabulary(story_list: list[str], tokenizer=SpacyTokenizer(),
                      max_words: int | None = None) -> dict[str, int]:
    """
    Creates a vocabulary (a mapping of string-tokens to indices) from a list of stories, containing the max_words-4 most
    frequent words. Also adds <eos>, <bos>, <pad> and <unk> tokens to the vocabulary.
    :param story_list: The list of stores (e.g., the training dataset).
    :param tokenizer: The tokenizer for tokenizing the story.
    :param max_words: The maximum number of words that the vocabulary may contain.
    :return: The created vocabulary.
    """
    vocab_freq = get_token_frequencies(story_list, tokenizer)
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

    # handle capitalization after closing quotes for hardcoded cases
    speech = ['said', 'explained', 'replied', 'responded', 'answered', 'shouted', 'whispered', 'called', 'asked', 'cried']
    speech += [word.capitalize() for word in speech]

    pattern = re.compile(r'",?\s+[\w ]*(' + "|".join(speech) + r')[\w ]*[^\w\s]')

    def lower_after_speech(match):
        if re.search(r'"\s+The', match.group(0)):
            if re.search(r'"\s+[\w ]*(' + "|".join(speech) + r')\.', match.group(0)):
                pass
            else:
                return match.group(0)
        sub_match = re.search(r'",?\s+(\w)', match.group(0))
        if sub_match:
            lowered = sub_match.group(1).lower()
            return match.group(0).replace(sub_match.group(1), lowered)
        return match.group(0)

    story = pattern.sub(lower_after_speech, story)

    # names obtained from GPT-4o by providing list of vocabulary and asking for names:
    names = {'alice', 'amy', 'anna', 'ben', 'bella', 'benny', 'billy', 'bob', 'bobo', 'daisy', 'dave', 'emily',
             'emma', 'ellie', 'ella', 'george', 'jack', 'jake', 'jane', 'jen', 'jenny', 'jim', 'jimmy', 'joe',
             'john', 'johnny', 'kitty', 'leo', 'lila', 'lily', 'lisa', 'lola', 'lucy', 'mandy', 'mark', 'mary', 'max', 'mia',
             'mike', 'molly', 'pete', 'peter', 'polly', 'rex', 'sally', 'sam', 'sammy', 'sara', 'sarah', 'sophie', 'sue',
             'susie', 'tim', 'timmy', 'tom', 'tommy', 'toby'}

    # replace names with capitalized names
    story = re.sub(r'\b(' + '|'.join(names) + r')\b', lambda x: x.group().capitalize(), story)

    story = re.sub(r'\bi\b', r'I', story)  # Fix capitalization of 'i'

    return story


def prompt_model(model_name: str, start_str: str, length: int = 250, temperature: float = 1.0, method: str = "default",
                 beam_width: int = 5, top_k: int = 30) -> str:
    """
    Generates a story continuation using a specified model and method.
    :param model_name: The name of the model to load (e.g., 'transformer_8.3M').
    :param start_str: The beginning of the story.
    :param length: The maximum length of the generated sequence (in tokens).
    :param temperature: The temperature used for sampling.
    :param method: The generation method. Options are 'default', 'beam', and 'beam_multinomial'.
    :param beam_width: The beam width for beam search methods.
    :param top_k: The number of top tokens to consider during multinomial sampling.
    :return: The generated story (as a string).
    """
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
    """
    Dataset class for handling the stories from the TinyStories dataset. This class provides the functionality to
    load stories, to tokenize them and to create a batch for training/evaluation.
    """
    def __init__(self, vocabulary: dict, tokenizer=SpacyTokenizer(), split: str = "train",
                 dataset_path: str = "data/TinyStoriesV2", max_seq_len: int | None = None):
        """
        :param vocabulary: The vocabulary (a dictionary containing the mapping of token-strings to token-indices).
        :param tokenizer: A tokenizer function.
        :param split: The dataset split to use ("train", "validation", "test").
        :param dataset_path: Path to the dataset directory.
        :param max_seq_len: The maximum sequence length.
        """
        self.stories = load_from_disk(dataset_path)[split]["text"]
        self.vocab = vocabulary
        self.tokenizer = tokenizer

        self.unk_token = self.vocab["<unk>"]
        self.pad_token = self.vocab["<pad>"]
        self.eos_token = self.vocab.get("<eos>")
        self.bos_token = self.vocab.get("<bos>")
        self.max_seq_len = max_seq_len if max_seq_len is not None else 8192

        # only relevant for old versions of the vocabulary
        if self.eos_token is None:
            raise ValueError("<eos> token is not found in the vocabulary.")
        elif self.bos_token is None:
            raise ValueError("<bos> token is not found in the vocabulary.")

    def get_batch(self, sequences: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Creates a batch (in the batch_first format) from a list of stories. Shorter stories will be padded to and larger
        stories will be cut off after max_seq_len tokens.
        :param sequences: A list of Tensors (stories), each representing a sequence of token indices (i.e., the
         __getitem__ method has to be applied to the individual stories first).
        :return: A tuple containing
            - A Tensor of shape (batch_size, max_seq_len), representing a batch of stories (excl. the last token)
            - A Tensor shifted by one token of shape (batch_size, max_seq_len) (representing the 'reference story').
            - A Tensor of the length of each story in the batch.
        """
        lengths = torch.tensor([s.shape[0] - 1 for s in sequences])
        padded_seq = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token)
        return padded_seq[:, :-1].contiguous(), padded_seq[:, 1:].contiguous(), lengths

    def get_stories(self) -> list[str]:
        """
        :return: The list of stories, with which this class object has been initialized.
        """
        return self.stories

    def __getitem__(self, index: int) -> Tensor:
        """
        :param index: The index of the story to be retrieved.
        :return: A tensor of token indices representing the tokenized story.
        """
        # Note: The returned tensor will have a length up to max_seq_length+1, since the last/first token will be
        # stripped for 'training' / 'reference' Tensor.
        story = self.stories[index]
        tokens = self.tokenizer(story.lower())

        token_list = [self.bos_token]
        # append at most max_seq_len tokens to the token_list
        token_list.extend(self.vocab.get(token, self.unk_token)
                          for _, token in zip(range(self.max_seq_len), tokens))

        if len(token_list) <= self.max_seq_len:
            token_list.append(self.eos_token)

        return torch.tensor(token_list, dtype=torch.int64)

    def __len__(self) -> int:
        """
        :return: The number of stories in the dataset split
        """
        return len(self.stories)


def save_vocabulary(vocabulary: dict[str, int], file_path: str = "trained_models/vocabulary.pkl"):
    """
    Saves a vocabulary dictionary (as a specified pickle file) to disk.
    :param vocabulary: A vocabulary dictionary to save.
     A key of this vocabulary should represent a token-string, while the value should represent the corresponding index
     of the token (for the embedding).
    :param file_path: The path to the pickle file where the vocabulary will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(vocabulary, file)


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    abs_path = os.path.abspath(os.path.join(script_dir, relative_path))
    return abs_path


def load_vocabulary(file_path: str = "trained_models/vocabulary.pkl") -> dict[str, int]:
    """
    Loads a vocabulary dictionary (from a specified file). Each key of this vocabulary is a token-string, while the
    value represents the corresponding index (for the embedding).
    :param file_path: The path to the vocabulary file, stored in the pickle format.
    :return vocab: The loaded vocabulary dictionary.
    """
    abs_filename = get_absolute_path(file_path)
    if not os.path.exists(abs_filename):
        raise FileNotFoundError(f"Vocabulary file not found: {abs_filename}")
    with open(abs_filename, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Displays the current vocabulary
    loaded_vocab = load_vocabulary("trained_models/vocabulary.pkl")
    print(f"Vocabulary:\n{loaded_vocab}")
