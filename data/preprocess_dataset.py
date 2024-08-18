import os
import sys
import random
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

replacement_table = {
    '\u200b': '',
    '\xa0': '',
    '\xad': '',
    '…': '...',
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    ' – ': ' – ',
    '—': ' – ',
    '\n': ' ',
    '”': '"',
    '“': '"'
}
allowed_non_ascii_symbols = {'–'}


def find_non_ascii_symbols(story: str) -> bool:
    for ch in story:
        if ch.isascii() is False and ch not in allowed_non_ascii_symbols:
            return True
    return False


def clean_dataset(stories: list[str] | set[str], is_test_split: bool = False, min_length: int = 180) -> list[str]:
    """
    :param stories: list/set of the TinyStories
    :param is_test_split: In the test set, no story is removed
    :param min_length: minimum length of a story, in *characters*
    :return:
    """
    cleaned_stories = []
    for story in stories:
        for k, v in replacement_table.items():
            story = story.replace(k, v)
        if "  " in story or "*" in story:
            story = " ".join(story_part.strip("*") for story_part in story.split())
        story = story.strip()

        allowed_chars_only = (story.isascii() is True or find_non_ascii_symbols(story) is False)
        if is_test_split is True or (len(story) >= min_length and allowed_chars_only):
            cleaned_stories.append(story)

    return cleaned_stories


def load_raw_stories(filename: str) -> list[str]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"The file {filename} was not found!", file=sys.stderr)
        sys.exit(1)

    return data.split("<|endoftext|>")


def create_dataset(version: str = "v2", low_memory: bool = False, validation_size: int = 73728):
    if version.lower() == "v2":
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
        train_dataframe.to_csv(f"{dataset_name}-train.csv", index=False)
        val_dataframe.to_csv(f"{dataset_name}-valid.csv", index=False)
        test_dataframe.to_csv(f"{dataset_name}-test.csv", index=False)
        dataset_dict = load_dataset('csv',
                                    data_files={'train': "TinyStoriesV2-train.csv",
                                                'validation': "TinyStoriesV2-valid.csv",
                                                'test': "TinyStoriesV2-test.csv"})
    else:
        train_dataset = Dataset.from_pandas(train_dataframe)
        val_dataset = Dataset.from_pandas(val_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
    dataset_dict.save_to_disk(dataset_name)


if __name__ == '__main__':
    create_dataset()
