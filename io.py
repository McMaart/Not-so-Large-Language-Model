from datasets import load_dataset
import re


def load_tiny_stories(end: int, start: int = 0, split="train"):
    """
    (Down-)Loads the TinyStories Dataset and returns the entries 'start' to 'end - 1' from the chosen split
    (i.e., returns 'end - start' many stories)
    """
    return load_dataset("roneneldan/TinyStories")[split][start:end]['text']


def load_from_file(filename: str) -> list:
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read().split('\n<end>\n\n')[:-1]


def get_vocabulary(story_list: list) -> dict:
    vocabulary = {}
    for item in story_list:
        tokens = re.split(r'\b(?![ \n])', item.lower())[1:]
        for token in tokens:
            token = token.strip('"').strip("'")
            token = token.strip()
            vocabulary.setdefault(token, 0)
            vocabulary[token] += 1
    return vocabulary


def tokens_to_story(token_list: list) -> str:
    # ToDo: Remove whitespace after punctuation
    return " ".join(token_list)


def save_to_file(filename: str, story_list: list):
    with open(filename, 'w', encoding="utf-8") as f:
        for item in story_list:
            item = "\n".join(item.split("\n\n"))
            f.write(f"{item}\n<end>\n\n")


if __name__ == "__main__":
    # stories = load_tiny_stories(100)
    # save_to_file("data/100stories.txt", stories)
    stories = load_from_file("data/100stories.txt")
    token_dict = get_vocabulary(stories)
    print("Stories:", stories)
    print(f"Number of tokens: {len(token_dict)}")
    print("Token frequency:", {k: v for k, v in sorted(token_dict.items(), key=lambda item: item[1], reverse=True)})
