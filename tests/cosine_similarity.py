from datasets import load_from_disk
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    generated_story = """"
Once upon a time, there was a little boy named Timmy. Timmy loved to play outside with his friends. One day, Timmy's mom asked him to help her. Timmy was excited to help her, so his mom helped him. They went to the park and Timmy saw a big tree with lots of leaves. He wanted to climb the tree, but his mom said no. Timmy was sad and started to cry. His mom asked him if he could climb the tree, but Timmy said no. He was sad and frustrated. Suddenly, Timmy heard a loud noise outside. He looked around and saw a big dog. The dog looked scared and didn't know what to do. Timmy's mom said, "Don't worry, we can help you. We can help you." Timmy was relieved and thanked the dog. He went back to his friends and said, "Thank you for helping me, Timmy. You are so kind." <eos>
    """

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

