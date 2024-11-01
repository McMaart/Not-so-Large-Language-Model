from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(generated_story: str, reference_stories: list[str], tokenizer=None):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    max_sim = 0
    max_idx = 0

    for i, story in enumerate(reference_stories):
        tfidf_matrix = vectorizer.fit_transform([story, generated_story])
        cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        if cos_sim > max_sim:
            max_sim = cos_sim
            max_idx = i
            print(f"Index: {max_idx}, Similarity: {round(max_sim, 6)}\nStory: {story}", end="\n\n")
    return max_sim, max_idx


if __name__ == '__main__':
    dataset = load_from_disk("../data/TinyStoriesV2")["train"]["text"]
    sample_story = """Once upon a time, there was a small hill. On top of the hill, there was a white house. In this 
    house lived a kind doctor who loved to help others. One day, a little girl named Lucy came to the doctor. She 
    fell down the hill and hurt her knee. The doctor was very kind and wanted to help her. He put a band - hop on 
    Lucy's knee and gave her a hug. Lucy felt better and said, "Thank you, doctor, for helping me." The doctor smiled 
    and waved goodbye. Lucy went back to the white house, feeling safe and happy on the hill. From that day on, 
    Lucy would visit the white hospital and play there."""
    cosine_sim(sample_story, dataset)
