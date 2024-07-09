import random
import pandas as pd
from datasets import Dataset, DatasetDict

# using Python 3.7+ (else, use an OrderedDict instead)
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
        if "  " in story or "*" in story:  # ToDo: also remove _
            story = " ".join(story_part.strip("*") for story_part in story.split())

        allowed_chars_only = (story.isascii() is True or find_non_ascii_symbols(story) is False)
        if is_test_split is True or (len(story) >= min_length and allowed_chars_only):
            cleaned_stories.append(story)

    return cleaned_stories


if __name__ == '__main__':
    with open("TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        data = f.read()
    story_list = [story.strip() for story in data.split("<|endoftext|>")[:-1]]
    story_set = set(story_list)

    initial_story_count = len(story_list)
    print(f"Initial number of stories: {initial_story_count}")

    story_set.discard("")
    train_and_val_stories = clean_dataset(story_set)

    cleaned_story_count = len(train_and_val_stories)
    print(f"Number of stories after cleaning: {cleaned_story_count}")

    validation_size = 73728
    random.seed(42)
    random.shuffle(train_and_val_stories)
    train_stories = train_and_val_stories[:-validation_size]
    val_stories = train_and_val_stories[-validation_size:]
    train_stories.sort(key=lambda s: len(s))
    val_stories.sort(key=lambda s: len(s))

    print(f"Number of training stories: {len(train_stories)}")
    print(f"Number of validation stories: {len(val_stories)}")

    # shortest_train_story_length = min(len(story) for story in train_stories)
    # print(f"Length of the shortest training story after cleaning: {shortest_train_story_length}")

    with open("TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
        data = f.read()
    story_list = [story.strip() for story in data.split("<|endoftext|>")[1:]]
    test_stories = clean_dataset(story_list, is_test_split=True)
    print(f"Number of test stories: {len(test_stories)}")

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_stories, columns=["text"]))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_stories, columns=["text"]))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_stories, columns=["text"]))

    dataset_name = "TinyStories"
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    dataset_dict.save_to_disk(dataset_name)
