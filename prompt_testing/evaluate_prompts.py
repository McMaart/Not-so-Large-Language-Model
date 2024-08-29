from parse_model_reply import parse_model_reply, categories
from evaluation.ollama_eval import get_llama_response
from sklearn.metrics import mean_squared_error
import numpy as np

def load_prompts() -> list[str]:
    with open("prompts.txt", "r") as f:
        prompts = f.read().split("#PROMPT#SEPERATOR#")
        for i in range(len(prompts)):
            prompts[i] = prompts[i].strip()
            prompts[i] = "\n".join(prompts[i].split("\n")[1:])
    return prompts

def load_stories_and_ratings() -> list[tuple[str, dict[str, list[int]]]]:
    story_w_rating = []

    with open("stories.txt", "r") as f:
        stories_w_ratings = f.read().split("#STORY#SEPERATOR#")

        for story_w_rating in stories_w_ratings:
            rating = {categories[i]: [] for i in range(len(categories))}

            story, ratings = story_w_rating.split("#STORY#RATINGS#SEPERATOR#")

            story = story.strip()
            ratings = ratings.strip()
            ratings = ratings.split("\n")

            for line in ratings:
              for category in categories.values():
                  if line.startswith(category + ":"):
                      rating[category].append(int(line.split(":")[1].strip()))

            story_w_rating.append((story, rating))
    return story_w_rating

def evaluate_prompts():
    prompts = load_prompts()
    story_w_rating = load_stories_and_ratings()
    true_ratings = [] # n_stories X n_categories X 3
    predicted_ratings = [] # n_stories X n_prompts X n_categories

    for story_idx, (story, rating) in enumerate(story_w_rating):
        true_ratings.append(list(rating.values()))
        predicted_ratings.append([])
        for prompt_idx, prompt in enumerate(prompts):
            prompt = prompt.replace("#STORY#PLACEHOLDER#", story)
            reply = get_llama_response(prompt)
            parsed_ratings = parse_model_reply(reply, prompt_idx)
            predicted_ratings[story_idx].append(parsed_ratings)

    n_stories = len(true_ratings)
    n_prompts = len(prompts)
    n_categories = len(categories)

    avg_rmse = np.zeros((n_prompts, n_categories))
    best_prompts = np.zeros(n_categories, dtype=int)

    for story_idx in range(n_stories):
        for prompt_idx in range(n_prompts):
            for category_idx in range(n_categories):
                # This line might me wrong:
                # avg_rmse[prompt_idx, category_idx] += mean_squared_error(true_ratings[story_idx][category_idx], predicted_ratings[story_idx][prompt_idx][category_idx], squared=False)
                for rating_idx in range(3):  
                    avg_rmse[prompt_idx, category_idx] += mean_squared_error([true_ratings[story_idx][category_idx][rating_idx]], [predicted_ratings[story_idx][prompt_idx][category_idx]], squared=False)

    avg_rmse /= (n_stories * 3)  # Divide by n_stories * 3 to get average RMSE

    for category_idx in range(n_categories):
        best_prompts[category_idx] = np.argmin(avg_rmse[:, category_idx])

    print("Average RMSE for each category:")
    for i, category in categories.items():
        print(f"{category}: {avg_rmse[:, i]}")
    print()
    print("Best prompt for each category:")
    for i, category in categories.items():
        print(f"{category}: {prompts[best_prompts[i]]}")
    print("Overall best prompt:")
    print(prompts[np.argmin(avg_rmse.sum(axis=1))])
      

if __name__ == "__main__":
    evaluate_prompts()



