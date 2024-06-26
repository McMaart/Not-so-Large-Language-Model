from datasets import load_from_disk
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    generated_story = """Once upon a time, there was a little girl named Lily. She loved to play outside and explore the 
    world around her. One day, she found a shiny rock on the ground. She picked it up and carried it to her mommy. 
    "Mommy, look what I found!" She said. "I found it!" Her mommy smiled and said, "That's a nice rock, Lily. But 
    remember, it's important to be honest and not be selfish." Lily nodded and continued to rock the rock until she was 
    happy again."""

    # nlp = spacy.load("en_core_web_sm")
    # s1 = nlp(generated_story)
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    dataset = load_from_disk("../data/TinyStories")
    max_sim = 0
    max_idx = 0

    for i, item in enumerate(dataset["train"]):
        # if i % 2000 == 0:
        #     print(i, max_sim, max_idx)
        story = item["text"]
        # cosine_sim = s1.similarity(nlp(story))
        tfidf_matrix = vectorizer.fit_transform([story, generated_story])
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        if cosine_sim > max_sim:
            max_sim = cosine_sim
            max_idx = i
            print(f"Index: {max_idx}, Similarity: {max_sim}, Story: {story}", end="\n\n")
