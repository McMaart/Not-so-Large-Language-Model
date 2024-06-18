from parse_model_reply import parse_model_reply, categories
from tests.ollama_eval import get_llama_response
from sklearn.metrics import mean_squared_error
import numpy as np

def load_prompts() -> list[str]:
    with open("prompts.txt", "r") as f:
        prompts = f.read().split("#PROMPT#SEPERATOR#")
        for i in range(len(prompts)):
            prompts[i] = prompts[i].strip()
            prompts[i] = "\n".join(prompts[i].split("\n")[1:])
    return prompts

def load_stories_and_ratings() -> list[tuple[str, dict[str, float]]]:
    story_w_ratings = []

    with open("stories.txt", "r") as f:
        stories_with_ratings = f.read().split("#STORY#SEPERATOR#")

        for story_with_rating in stories_with_ratings:
            rating = {category: [] for category in categories.values()}

            try:
                story, ratings = story_with_rating.split("#STORY#RATINGS#SEPERATOR#")
            except ValueError:
                continue  # Skip if the format is incorrect

            story = story.strip()
            ratings = ratings.strip().split("\n")

            for line in ratings:
                for category in categories.values():
                    if line.startswith(category + ":"):
                        value = line.split(":")[1].strip()
                        if value.isdigit():
                            rating[category].append(int(value))
                        else:
                            rating[category].append(None)  # Handle non-digit ratings as None

            # Calculate the mean ratings for each category
            mean_rating = calculate_mean_ratings(rating)

            story_w_ratings.append((story, mean_rating))

    return story_w_ratings

def calculate_mean_ratings(rating: dict[str, list[int]]) -> dict[str, float]:
    """
    Calculate the mean ratings for each category.

    Parameters:
    rating (dict): Dictionary containing lists of ratings for each category.

    Returns:
    dict: Dictionary containing mean ratings for each category.
    """
    mean_rating = {}
    for category, values in rating.items():
        valid_values = [value for value in values if value is not None]
        if valid_values:
            mean_rating[category] = sum(valid_values) / len(valid_values)
        else:
            mean_rating[category] = None  # Handle missing ratings by setting to None
    return mean_rating

def evaluate_prompts():
    prompts = load_prompts()
    story_w_rating = load_stories_and_ratings()
    true_ratings = []  # n_stories X n_categories
    predicted_ratings = []  # n_stories X n_prompts X n_categories
    failed_prompts = [0] * len(prompts)
    failed_prompt_details = {i: [] for i in range(len(prompts))}  # To store details of failures

    for story_idx, (story, mean_rating) in enumerate(story_w_rating):
        true_ratings.append(mean_rating)
        predicted_ratings.append([])
        for prompt_idx, prompt in enumerate(prompts):
            prompt = prompt.replace("#STORY#PLACEHOLDER#", story)
            reply = get_llama_response(prompt)
            print(f"\n REPLY: {reply}")
            try:
                parsed_ratings = parse_model_reply(reply, prompt_idx)
                if parsed_ratings is None or any(rating is None for rating in parsed_ratings):
                    raise ValueError(f"Failed to parse ratings for prompt index {prompt_idx}")
                predicted_ratings[story_idx].append(parsed_ratings)
            except Exception as e:
                print(f"Error parsing prompt index {prompt_idx}: {e}")
                parsed_ratings = [None] * len(categories)  # Append a placeholder with None
                predicted_ratings[story_idx].append(parsed_ratings)
                failed_prompts[prompt_idx] += 1  # Increment failure count
                failed_prompt_details[prompt_idx].append(story)  # Store failed story

        print(f"predicted_ratings: {predicted_ratings}")

    n_stories = len(true_ratings)
    valid_prompts = [i for i, count in enumerate(failed_prompts) if count == 0]

    if not valid_prompts:
        print("No valid prompts found.")
        return

    n_categories = len(categories)

    # Calculate RMSE for valid prompts only
    avg_rmse = np.zeros((len(valid_prompts), n_categories))
    best_prompts = np.zeros(len(categories), dtype=int)

    for story_idx in range(n_stories):
        for valid_prompt_idx, prompt_idx in enumerate(valid_prompts):
            for category_idx, category in enumerate(categories.values()):
                true_value = true_ratings[story_idx][category]
                predicted_value = predicted_ratings[story_idx][prompt_idx][category_idx]

                if true_value is not None and predicted_value is not None:
                    avg_rmse[valid_prompt_idx, category_idx] += mean_squared_error(
                        [true_value],
                        [predicted_value],
                        squared=False  # Root mean squared error
                    )

    avg_rmse /= n_stories  # Divide by n_stories to get average RMSE

    for category_idx in range(n_categories):
        best_prompts[category_idx] = valid_prompts[np.argmin(avg_rmse[:, category_idx])]

    print("Average RMSE for each category:")
    for i, category in categories.items():
        print(f"{category}: {avg_rmse[:, i]}")
    print()
    print("Best prompt for each category:")
    for i, category in categories.items():
        print(f"Best prompt for {category}: {prompts[best_prompts[i]]}")

    print("\n PROMPTS EXCLUDED FOR FAILURES:")
    for i, count in enumerate(failed_prompts):
        if count > 0:
            print(f"Prompt {i}: {prompts[i]} (Failed {count} times)")
            for story in failed_prompt_details[i]:
                print(f"  FAILED STORY: {story}")
    print()
    print("\n Overall best prompt:")
    print(prompts[valid_prompts[np.argmin(avg_rmse.sum(axis=1))]])

if __name__ == "__main__":
    evaluate_prompts()
