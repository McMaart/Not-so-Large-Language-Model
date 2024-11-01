from io_utils import prompt_model


def generate_multiple_stories(model_name: str, start_str: str, n: int = 5, length: int = 250, temperature: float = 1.0,
                              method: str = "default", beam_width: int = 5, top_k: int = 30) -> list:
    """
    Generates n stories using the specified model and returns them in a list.

    :param model_name: Name of the model to use for story generation
    :param start_str: Starting string for the stories
    :param n: Number of stories to generate
    :param length: Length of each story
    :param temperature: Sampling temperature
    :param method: Generation method ('default', 'beam', 'beam_multinomial')
    :param beam_width: Beam width for beam search (if method is 'beam' or 'beam_multinomial')
    :param top_k: Top-k sampling for beam_multinomial method
    :return: List of generated stories
    """
    stories = []
    for _ in range(n):
        story = prompt_model(model_name, start_str, length, temperature, method, beam_width, top_k)
        stories.append(story)
    return stories


prompt_v1 = """
The following story is a story written for children at the age of around 5 years old.
Your task is to rate the story objectively.

Story:
{}

Please rate the story in the following categories:
- GRAMMAR (exclusively rate the grammar of the story)
- SPELLING (exclusively rate the spelling of the story)
- CONSISTENCY (exclusively rate the consistency of the story)
- STORY (exclusively rate the story/plot of the story)
- CREATIVITY (exclusively rate the creativity of the story)
- STYLE (exclusively rate the linguistic style of the story)

Rate objectively and rate each category independently from the others.
Rate each category with a score from 1 to 10, with 1 being the worst and 10 being the best.

It is crucial that your response ends with the following format (substitute <YOUR_CATEGORY_SCORE> with the score you gave for the respective category) and does not include any other text afterwords:

GRAMMAR: <YOUR_GRAMMAR_SCORE>
SPELLING: <YOUR_SPELLING_SCORE>
CONSISTENCY: <YOUR_CONSISTENCY_SCORE>
STORY: <YOUR_STORY_SCORE>
CREATIVITY: <YOUR_CREATIVITY_SCORE>
STYLE: <YOUR_STYLE_SCORE>
"""

prompt_v2 = """
The following story is a story written for children at the age of around 5 years old. Your task is to rate the story objectively.

    Story:
    {}

Please rate the story in the following categories:

GRAMMAR (exclusively rate the grammar of the story)
SPELLING (exclusively rate the spelling of the story)
CONSISTENCY (exclusively rate the consistency of the story)
STORY (exclusively rate the story/plot of the story)
CREATIVITY (exclusively rate the creativity of the story)
STYLE (exclusively rate the linguistic style of the story)
Rate objectively and rate each category independently from the others. Rate each category with a score from 1 to 10, with 1 being the worst and 10 being the best.

Important Instructions:

Ensure that your ratings for each category are based solely on the criteria specified for that category.
End your response with the format specified below.
Do not include any text other than the ratings after the specified format.
Ensure consistency and objectivity in your ratings.
Examples:

Example 1:

Story:
"Tommy the turtle walked slowly to the pond. He met a friendly frog named Fred. Together, they swam and played games until the sun set. They found a hidden treasure chest by the pond. Inside were shiny stones and colorful shells. They promised to meet again the next day to play with their new treasures."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 10
STORY: 8
CREATIVITY: 7
STYLE: 9

Example 2:

Story:
"A dragon flyed over the castle. The prince lookd up and waved. He wished he could fly too. The dragon landed and offered him a ride. They flew over the kingdom and saw many amazing sights. The prince felt like he was in a dream."

Ratings:
GRAMMAR: 7
SPELLING:7
CONSISTENCY: 9
STORY: 5
CREATIVITY: 6
STYLE: 5

Example 3:

Story:
"Once upon a time, there was a little mouse. The mouse loved cheese. One day, he found a big piece of cheese in a trap. He cleverly used a stick to get it out without getting caught. Then he shared the cheese with his friends. They had a big cheese party and danced all night."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 10
STORY: 6
CREATIVITY: 5
STYLE: 6

Example 4:

Story:
"The stars shone brightly in the sky. It was a magical night. Lucy made a wish on a shooting star, hoping for a new puppy. The next morning, her wish came true when she found a puppy in her backyard. She named him Sparky. They became best friends and went on many adventures together."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 9
STORY: 7
CREATIVITY: 7
STYLE: 8

Example 5:

Story:
"The robot walkeded into the room. It said, 'Hello, human!' Timmy was amazed and excited to meet a real robot. They played together all day and became best friends. The robot taught Timmy many things about space. At the end of the day, the robot had to leave, but it promised to come back."

Ratings:
GRAMMAR: 8
SPELLING: 5
CONSISTENCY: 8
STORY: 5
CREATIVITY: 7
STYLE: 5

Example 6:

Story:
"A magical unicorn danced under the rainbow. Everyone was amazed by its beauty. The unicorn spread joy and happiness wherever it went, making everyone smile. One day, it met a sad little girl. The unicorn took her on a magical ride over the rainbow. The girl was happy and never felt sad again."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 9
STORY: 7
CREATIVITY: 8
STYLE: 8

Example 7:

Story:
"Once upon a time, there was a cat. The cat chased a mouse and caught it. But instead of eating it, the cat befriended the mouse, and they had many adventures together. They explored the fields and forests. They even helped other animals in need. Their friendship was known far and wide."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 10
STORY: 6
CREATIVITY: 5
STYLE: 6

Example 8:

Story:
"There was a little girl named Sue. She loved to play with her doll. One day, her doll came to life, and they went on a magical adventure in a land of fairies and talking animals. They helped the fairies solve problems and made many new friends. Sue wished she could stay forever. But she knew she had to return home."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 9
STORY: 7
CREATIVITY: 8
STYLE: 8

Example 9:

Story:
"The spaceship zoomed through the galaxy. Aliens waved from their planets. The astronauts waved back, excited to explore new worlds and make new friends. They landed on a strange planet with giant flowers. They discovered a new species of friendly aliens. The astronauts and aliens shared stories and games."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 9
STORY: 7
CREATIVITY: 8
STYLE: 7

Example 10:

Story:
"The boy runned fast to catch the ball. He was happy when he caught it. His friends cheered, and they all continued to play happily in the park. Suddenly, it started to rain. They ran to take shelter under a big tree. They laughed and waited for the rain to stop."

Ratings:
GRAMMAR: 7
SPELLING: 10
CONSISTENCY: 9
STORY: 6
CREATIVITY: 4
STYLE: 5

Example 11:

Story:
"Once there was a brave knight. He rode his horse through the dark forest. He fought off bandits and saved villagers. One day, he met a wizard who gave him a magical sword. The knight used the sword to bake a cake. Then he flew to the moon and decided to become a farmer. The villagers never saw him again."

Ratings:
GRAMMAR: 10
SPELLING: 10
CONSISTENCY: 2
STORY: 4
CREATIVITY: 5
STYLE: 7

Format to Follow:
GRAMMAR: <YOUR_GRAMMAR_SCORE>
SPELLING: <YOUR_SPELLING_SCORE>
CONSISTENCY: <YOUR_CONSISTENCY_SCORE>
STORY: <YOUR_STORY_SCORE>
CREATIVITY: <YOUR_CREATIVITY_SCORE>
STYLE: <YOUR_STYLE_SCORE>

Remember: Ensure there is no additional text, explanations, or comments after the scores. """


if __name__ == '__main__':
    from evaluation.uhhgpt_selenium import webdriver_setup, get_ratings
    import numpy as np
    np.set_printoptions(precision=4)

    model_name = "35M"  # Example model name
    start_str = ""
    n = 100  # Number of stories to generate
    length = 255
    temperature = 0.7
    method = "default"  # Change to 'beam' or 'beam_multinomial' if needed
    beam_width = 10  # if beam
    top_k = 25  # if beam_multinomial

    stories = generate_multiple_stories(model_name, start_str, n=n, length=length, temperature=temperature, method=method)

    driver = webdriver_setup()
    ratings = get_ratings(driver, prompt_v1, stories)
    print(f"Rating list: {ratings}")

    rating_arr = np.array(ratings)
    if None not in rating_arr:
        print(f"Avg. rating: {rating_arr.mean(axis=0)}")
        print(f"Std:         {rating_arr.std(axis=0)}")
