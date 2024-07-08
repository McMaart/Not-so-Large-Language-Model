import os
import time
import numpy as np
import random
from datasets import load_from_disk
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io_utils import prompt_model
import torch
from torchtext.data import get_tokenizer

# Ensure the correct device is used (GPU if available, otherwise CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the maximum sequence length the model can handle
MAX_SEQUENCE_LENGTH = 256


def generate_story(prompt, model_name, method="default", max_length=250, temperature=1.0, beam_width=5, top_k=25):
    return prompt_model(model_name, prompt, length=max_length, temperature=temperature, method=method,
                        beam_width=beam_width, top_k=top_k)


def calculate_cosine_similarity(text1, text2, vectorizer):
    tfidf_matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_rouge(reference, generated):
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores


def evaluate_model(model_name, prompts, completions, entire_dataset, method="default", beam_width=5, top_k=50,
                   temperature=1.0):
    # Use the SpaCy tokenizer
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # Fit the vectorizer on the entire dataset
    print("Fitting vectorizer on the entire dataset...")
    start_fit_time = time.time()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer).fit(entire_dataset)
    end_fit_time = time.time()
    print(f"Vectorizer fitted in {end_fit_time - start_fit_time:.2f} seconds.")

    generated_stories = [
        generate_story(prompt, model_name, method, max_length=min(len(completion), MAX_SEQUENCE_LENGTH),
                       temperature=temperature, beam_width=beam_width, top_k=top_k)
        for prompt, completion in zip(prompts, completions)]

    # Extract only the generated completions part
    generated_completions = [story[len(prompt):] for story, prompt in zip(generated_stories, prompts)]

    cosims = []
    rouges = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for generated, reference_completion in zip(generated_completions, completions):
        max_cosim = 0
        max_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

        cosim = calculate_cosine_similarity(generated, reference_completion, vectorizer)
        scores = calculate_rouge(reference_completion, generated)

        if cosim > max_cosim:
            max_cosim = cosim

        for key in max_rouge.keys():
            if scores[key].fmeasure > max_rouge[key]:
                max_rouge[key] = scores[key].fmeasure

        cosims.append(max_cosim)
        for key in max_rouge.keys():
            rouges[key].append(max_rouge[key])

    avg_cosim = np.mean(cosims)
    std_cosim = np.std(cosims)
    avg_rouge = {key: np.mean(rouges[key]) for key in rouges}
    std_rouge = {key: np.std(rouges[key]) for key in rouges}

    return avg_cosim, std_cosim, avg_rouge, std_rouge


def save_scores_to_file(model_name, avg_cosim, std_cosim, avg_rouge, std_rouge, method, temperature, num_stories):
    filename = f"{model_name}_{method}_cosine_rouge_scores.txt"
    with open(filename, 'a') as f:  # Changed from 'w' to 'a' to append results
        f.write("\n")
        f.write(f"Number of Stories: {num_stories}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Average Cosine Similarity: {avg_cosim}\n")
        f.write(f"Standard Deviation Cosine Similarity: {std_cosim}\n")
        f.write("Average ROUGE Scores:\n")
        for key, value in avg_rouge.items():
            f.write(f"  {key}: {value}\n")
        f.write("Standard Deviation ROUGE Scores:\n")
        for key, value in std_rouge.items():
            f.write(f"  {key}: {value}\n")
    print(f"Scores saved to {filename}")


def main():
    model_name = "1M"  # Replace with your actual model name
    dataset_path = 'data/TinyStories'  # Path to the dataset
    method = ""  # Choose the generation method: default, beam, beam_multinomial
    temperature = 0.5  # Set the temperature for generation
    beam_width = 5  # Set the beam width for beam search
    top_k = 25  # Set the top_k for beam_multinomial

    num_stories = 100

    # Load the dataset
    dataset = load_from_disk(dataset_path)['train']['text']

    # Randomly select num_stories from the dataset
    selected_indices = random.sample(range(len(dataset)), num_stories)
    selected_stories = [dataset[i] for i in selected_indices]

    # Truncate stories to 40% for prompts and get the completions
    prompts = [story[:int(0.4 * len(story))] for story in selected_stories]
    completions = [story[int(0.4 * len(story)):] for story in selected_stories]

    start_time = time.time()
    avg_cosim, std_cosim, avg_rouge, std_rouge = evaluate_model(model_name, prompts, completions, dataset, method,
                                                                beam_width, top_k, temperature)
    end_time = time.time()

    print(f"Execution Time: {end_time - start_time} seconds")
    print(f"Average Cosine Similarity: {avg_cosim}")
    print(f"Standard Deviation Cosine Similarity: {std_cosim}")
    print("Average ROUGE Scores:")
    for key, value in avg_rouge.items():
        print(f"  {key}: {value}")
    print("Standard Deviation ROUGE Scores:")
    for key, value in std_rouge.items():
        print(f"  {key}: {value}")

    # Save the scores to a file
    save_scores_to_file(model_name, avg_cosim, std_cosim, avg_rouge, std_rouge, method, temperature, num_stories)


if __name__ == "__main__":
    main()
