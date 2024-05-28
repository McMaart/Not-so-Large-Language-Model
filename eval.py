from rouge import Rouge
from io_utils import load_tiny_stories, clean_stories
from datasets import load_dataset


def calculate_rouge_scores(predicted_stories, actual_stories):
  rouge = Rouge(max_n=2)

  scores = rouge.get_scores(predicted_stories, actual_stories, avg=True)

  print(f"ROUGE-1: {scores['rouge-1']}")
  print(f"ROUGE-2: {scores['rouge-2']}")
  print(f"ROUGE-L: {scores['rouge-l']}")

  return scores

def max_rouge_score(pred_story: str, prompt: str):
  '''
  Returns the maximum ROUGE score for a given story
  '''
  # stories = [story for story in load_dataset("roneneldan/TinyStories")['train'][:1000]['text'] if story.lower().startswith(prompt.lower())]
  # stories = clean_stories(stories)

  '''
  Here, a larger slice than just the prompt is used to filter the stories so that the rouge calculation 
  finishes in a reasonable amount of time
  '''
  pred_start = pred_start = " ".join(pred_story.split()[:15])
  stories = clean_stories(load_dataset("roneneldan/TinyStories")['train'][:175000]['text'])
  print("Stories loaded and cleaned")
  stories = [story for story in stories if story[:len(pred_start)].lower() == pred_start.lower()]
  print("Found {} stories with same start".format(len(stories)))

  rouge = Rouge()
  max = 0
  most_similar = ""
  for story in stories:
    scores = rouge.get_scores(pred_story, story)
    if scores[0]['rouge-2']['f'] > max:
      max = scores[0]['rouge-2']['f']
      most_similar = story
  return max, most_similar

def get_stories(story_prompt: str, end: int, start: int = 0, split="train"):
  '''
  Gets stories that start with a given prompt
  '''
  stories = load_tiny_stories(end, start, split)
  stories = clean_stories(stories)
  stories = [story for story in stories if story.lower().startswith(story_prompt.lower())]
  return stories