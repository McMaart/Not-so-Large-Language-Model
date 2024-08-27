"""
File for cleaning up the 'raw' dataset (available at https://huggingface.co/datasets/roneneldan/TinyStories/tree/main),
and creating training, validation and test split (ref. Subsection 2.2 in our project report).
"""
import os
import sys
import random
from typing import Iterable
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# Note: Since Python 3.7, the insertion order is preserved
replacement_table = {
    '\u200b': '',  # Zero-width space
    '\xa0': '',  # Non-breaking space
    '\xad': '',  # Soft hyphen
    '\\': '',
    '*': '',
    # unify representation of quotation marks
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '”': '"',
    '“': '"',
    '\n': ' ',
    # unify symbol representing a break of thought
    '---': ' — ',
    '--': ' — ',
    ' - ': ' — ',
    '–': ' — ',
    ',"': ', "',
    '…': '...',
}
allowed_non_ascii_symbols = {'—'}


def find_non_ascii_symbols(story: str) -> bool:
    """
    Checks whether a story contains non-ASCII symbols that are not explicitly allowed.
    :param story: The input story to be checked.
    :return: True if the story contains not allowed non-ASCII symbols, False otherwise.
    """
    for ch in story:
        if ch.isascii() is False and ch not in allowed_non_ascii_symbols:
            return True
    return False


def clean_dataset(stories: Iterable[str], is_test_split: bool = False, min_length: int = 180) -> list[str]:
    """
    Cleans a list of stories by filtering out or replacing invalid characters or stories, as well as
    filtering out too short stories.
    :param stories: Iterable of stories to be cleaned.
    :param is_test_split: If True, no stories will be removed based on its length or character content.
    :param min_length: Minimum length of a story, in *characters*.
    :return: A list of the cleaned stories
    """
    cleaned_stories = []
    for story in stories:
        for k, v in replacement_table.items():
            story = story.replace(k, v)
        if "  " in story:
            story = " ".join(story_part for story_part in story.split())
        story = story.strip()

        allowed_chars_only = (story.isascii() is True or find_non_ascii_symbols(story) is False)
        if is_test_split is True or (len(story) >= min_length and allowed_chars_only):
            cleaned_stories.append(story)

    return cleaned_stories


def load_raw_stories(filename: str) -> list[str]:
    """
    Loads stories from a text file, which are seperated with an '<|endoftext|>' token.
    :param filename: The path to the text file.
    :return: A list of the stories.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"The file {filename} was not found!", file=sys.stderr)
        sys.exit(1)

    return data.split("<|endoftext|>")


def create_dataset(use_v2: bool = True, low_memory: bool = False, validation_size: int = 73728):
    """
    Given the raw story '-train' and '-valid' text files of a TinyStories dataset
     (available at https://huggingface.co/datasets/roneneldan/TinyStories/tree/main), applies certain cleaning steps
     to the stories and saves the preprocessed dataset in the Arrow IPC format.
    :param use_v2: If True, uses the TinyStoriesV2 dataset instead.
    :param low_memory: If True, saves the preprocessed stories as a csv file as an intermediate step. Recommended option
     if having 8GB of RAM or less.
    :param validation_size: The number of stories for the validation split
    """
    if use_v2 is True:
        dataset_name = "TinyStoriesV2"
        train_txt_file = "TinyStoriesV2-GPT4-train.txt"
        test_txt_file = "TinyStoriesV2-GPT4-valid.txt"
    else:
        dataset_name = "TinyStories"
        train_txt_file = "TinyStories-train.txt"
        test_txt_file = "TinyStories-valid.txt"

    if not os.path.isfile(test_txt_file):
        print(f"The file {test_txt_file} does not exist!", file=sys.stderr)
        sys.exit(1)

    # Create training and validation stories
    print("Loading the raw training and validation stories...")
    story_list = [story.strip() for story in load_raw_stories(train_txt_file)[:-1]]
    initial_story_count = len(story_list)
    print(f"Initial number of stories: {initial_story_count}")

    story_set = set(story_list)
    story_set.discard("")
    train_and_val_stories = clean_dataset(story_set)
    print(f"Number of stories after cleaning: {len(train_and_val_stories)}")

    random.seed(42)
    random.shuffle(train_and_val_stories)

    train_stories = train_and_val_stories[:-validation_size]
    val_stories = train_and_val_stories[-validation_size:]
    train_stories.sort(key=lambda s: len(s))
    val_stories.sort(key=lambda s: len(s))
    print(f"Number of training stories: {len(train_stories)}")
    print(f"Number of validation stories: {len(val_stories)}")

    train_dataframe = pd.DataFrame(train_stories, columns=["text"])
    val_dataframe = pd.DataFrame(val_stories, columns=["text"])

    # Create test stories
    story_list = [story.strip() for story in load_raw_stories(test_txt_file)[1:]]
    test_stories = clean_dataset(story_list, is_test_split=True)
    print(f"Number of test stories: {len(test_stories)}")
    test_dataframe = pd.DataFrame(test_stories, columns=["text"])

    if low_memory is True:
        data_files = {'train': f"{dataset_name}-train.csv",
                      'validation': f"{dataset_name}-valid.csv",
                      'test': f"{dataset_name}-test.csv"}
        train_dataframe.to_csv(data_files['train'], index=False)
        val_dataframe.to_csv(data_files['validation'], index=False)
        test_dataframe.to_csv(data_files['test'], index=False)
        dataset_dict = load_dataset('csv', data_files=data_files)
    else:
        train_dataset = Dataset.from_pandas(train_dataframe)
        val_dataset = Dataset.from_pandas(val_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    dataset_dict.save_to_disk(dataset_name)


if __name__ == '__main__':
    create_dataset()
