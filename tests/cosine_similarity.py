from datasets import load_from_disk
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    generated_story = """"
 There was once a little girl named Ella. Everyday Ella would look out her window and admire the beautiful, colorful world outside her window, it had so many things to see. One day, Ella saw something different in the sky. It was called a rainbow. It was so beautiful and made her very excited. She had never seen anything like it before! Ella was so happy to see the rainbow, and she waved to it. Then, she realized that the rainbow was too far away for her to see. Ella decided to try something new. She decided to build a bridge out of blocks. She found some sticks and stones and put it the way she wanted. She even used her special blocks to make a big bridge, with four small rocks. Ella was so proud of her new bridge. Her mom was happy too, because even though she was only three years old, Ella was determined to do something amazing.
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

