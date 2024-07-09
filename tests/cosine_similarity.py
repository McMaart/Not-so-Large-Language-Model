from datasets import load_from_disk
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    generated_story = """"
Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, she found a big box in her yard. She was very excited and wanted to find out what was inside. Lily went to her friend, Tom. "Tom, look at this box!" She said, holding the box. Tom looked at the box and said, "Wow, it's a big box!" They opened the box and found a lot of toys inside. Lily and Tom played with the toys all day. They had so much fun with the toys and the big box. At the end of the day, they were tired but happy. They sat down and ate the toys. They were glad they found the big box in the park. <eos>
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

