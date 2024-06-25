import re

categories = {
    0: "GRAMMAR",
    1: "SPELLING",
    2: "CONSISTENCY",
    3: "STORY",
    4: "CREATIVITY",
    5: "STYLE"
}


def parse_prompt_0(reply: str) -> list[int] | None:
    reply = reply.strip()
    lines = reply.split("\n")

    n_categories = len(categories)
    if len(lines) < n_categories:
        return None

    ratings = [None] * n_categories

    for line in lines[-n_categories:]:
        line = line.upper().strip()
        for i in range(n_categories):
            if line.startswith(categories[i] + ":"):
                try:
                    rating = int(line.split(":")[1].strip())
                    if 10 >= rating >= 1:
                        ratings[i] = rating
                    break
                except:
                    break
    return ratings


# Define additional parse functions if needed
def parse_prompt_4(reply: str) -> list[int] | None:
    reply = reply.strip()
    lines = reply.split("\n")

    n_categories = len(categories)
    if len(lines) < n_categories:
        return None

    ratings = [None] * n_categories

    for line in lines[-n_categories:]:
        line = line.upper().strip()
        for i in range(n_categories):
            if line.startswith(categories[i] + ":"):
                try:
                    rating = int(line.split(":")[1].strip())
                    if 10 >= rating >= 1:
                        ratings[i] = rating
                    break
                except ValueError:
                    break
    return ratings


def parse_prompt_3(reply: str) -> list[int] | None:
    ratings = [None] * len(categories)
    for i, category in categories.items():
        # Various patterns to match different formats of ratings
        pattern0 = rf"(?i){category.upper()}:.*? (\d+)"
        pattern1 = rf"(?i)\b{category}\b\s*[:\s]*\D*?(\d+)"
        pattern2 = rf"(?i)\b{category}\b\s*[:\s]*(N/A|None|No rating)"
        #pattern3 = rf"(?i)\b{category}\b\s*[:\s]*.*?\n.*?\bRating\b\s*[:\s]*(\d+)"
        #pattern4 = rf"(?i)\*\*{category}\*\*\s*[:\s]*.*?\n.*?(\d+)"
        #pattern5 = rf"(?i)\b{category}\b\s*[:\s]*.*?\n\s*(\d+)"
        #pattern6 = rf"(?i)\b{category}\b\s*[:\s]*.*?(\d+)"
        #pattern7 = rf"(?i)\b{category}\b\s*[:\s]*.*?\n.*?Ranking\b\s*[:\s]*(\d+)"
        #pattern8 = rf"(?i){category.upper()}[:\s]*\D*?(\d+)"  # Match CATEGORY: number or CATEGORY number
        #pattern9 = rf"(?i)\b{category}\b\s*.*?\s(\d+)"  # Match CATEGORY followed by any text and then a number
        #pattern10 = rf"(?i)\*\*{category}\*\*\s*.*?(\d+)"  # Match **CATEGORY** followed by a number
        #pattern11 = rf"(?i){category}\s*[:\s]*\n.*?(\d+)"  # Match CATEGORY followed by a newline and then a number
        #pattern12 = rf"(?i)\b{category}\b\s*[:\s]*.*?\n.*?\bRating\b\s*[:\s]*<(\d+)>"  # Match CATEGORY followed by Rating: <number>
        #pattern13 = rf"(?i)\b{category}\b\s*[:\s]*.*?\n\s*<(\d+)>"  # Match CATEGORY followed by a number in angle brackets
        #pattern14 = rf"(?i)\b{category}\b\s*[:\s]*\D*?(\d+)"  # Match CATEGORY followed by non-digits and then a number

        # Patterns to exclude numbers referring to ages
        # exclude_pattern = rf"(\d+)\s*(year|old)"

        match = (re.search(pattern0, reply) or re.search(pattern2, reply) or re.search(pattern1, reply))# or
                 #re.search(pattern3, reply)) #or re.search(pattern4, reply)) or re.search(pattern5, reply)) or
                 #re.search(pattern6, reply)) or re.search(pattern7, reply) or re.search(pattern8, reply) or
                 #re.search(pattern9, reply) or re.search(pattern10, reply) or re.search(pattern11, reply) or
                 #re.search(pattern12, reply) or re.search(pattern13, reply))

        if match:
            try:
                rating = int(match.group(1))
                if 10 >= rating >= 1:
                    ratings[i] = rating
            except ValueError:
                continue

    return ratings

parse_functions = {
    0: parse_prompt_3,
    1: parse_prompt_3,
    2: parse_prompt_3,
    3: parse_prompt_3,
    4: parse_prompt_3,
    5: parse_prompt_3,
    6: parse_prompt_3
}


def parse_model_reply(reply: str, prompt_idx: int) -> list[int] | None:
    parse_function = parse_functions.get(prompt_idx)
    if parse_function is None:
        raise ValueError(f"No parse function for prompt index {prompt_idx}")
    return parse_function(reply)



if __name__ == "__main__":
    reply1 = """
    Here are the ratings:

    **Grammar:** The story lacks complexity in sentence structure, but it is grammatically correct and free of errors.

    GRAMMAR: 8
    **SPELLING:** The story does not contain any spelling mistakes.
    SPELLING: 10
    **CONSISTENCY:** The story has a consistent tone and style throughout, although the plot is quite simple.
    CONSISTENCY: 6
    **STORY:** The story lacks a clear narrative or character development, but it may appeal to a young child's sense of humor.
    STORY: 4
    **CREATIVITY:** The story's use of a sudden ending may be humorous for some children, but it is not particularly creative.
    CREATIVITY: 3
    **STYLE:** The story has a playful tone and a simple, easy-to-understand style that may appeal to young children.
    STYLE: 7
    """

    reply2 = """
    Here are my ratings:

    GRAMMAR: 8
    The sentence is simple and easy to understand, but it could be improved with a subject-verb agreement (e.g., "There was nothing.").

    SPELLING: 10
    No spelling errors here!

    CONSISTENCY: 9
    The story stays consistent in its brevity and simplicity throughout.

    STORY: 3
    While the story is short, it's also a bit... well, boring. A 5-year-old might be expecting something more exciting or engaging.

    CREATIVITY: 2
    Unfortunately, this story doesn't really have any creative elements to speak of.

    STYLE: 4
    The writing style is straightforward and simple, but not particularly engaging or memorable.

    So, here are my ratings:

    GRAMMAR: 8
    SPELLING: 10 
    CONSISTENCY: 9
    STORY: 3
    CREATIVITY: 2
    STYLE: 4
    """

    reply3 = """
    Here are my ratings:

    GRAMMAR: 10
    SPELLING: 10
    CONSISTENCY: 5
    STORY: 1
    CREATIVITY: 8
    STYLE: 9

    Let me know if you need any further clarification!
    """

    reply4 = """
    Here are my ratings:

    **GRAMMAR:** The story only contains one sentence, which is grammatically correct. There's no complex sentence structure or punctuation issues. 9/10

    **SPELLING:** The story doesn't contain any spelling errors. All words are correctly spelled. 10/10

    **CONSISTENCY:** The story is consistent in its simplicity and brevity. It's a one-sentence story that doesn't try to be anything more complex. 6/10 (it could be more consistent in terms of plot or character development, but it's still a valid storytelling approach)

    **STORY:** The story lacks any real narrative or plot. It's essentially just a statement about the beginning of the universe, with no resolution or conclusion. 2/10

    **CREATIVITY:** While the idea of "nothing" is interesting, the execution is quite straightforward and doesn't bring anything new to the table. 4/10

    **STYLE:** The story has a very simple, straightforward style that's easy to understand. It's not trying to be poetic or elaborate. 7/10

    Here are my ratings in the required format:

    GRAMMAR: 9
    SPELLING: 10
    CONSISTENCY: 6
    STORY: 2
    CREATIVITY: 4
    STYLE: 7
    """

    reply5 = """
    Here are my ratings:

    GRAMMAR: 8
    SPELLING: 10
    CONSISTENCY: 9
    STORY: 2
    CREATIVITY: 4
    STYLE: 6

    Let me know if you'd like me to explain my reasoning for each category!
    """

    reply6 = """
        Here are my ratings:

        GRAMMAR: is good.
        Ranking: <10>
        SPELLING: few mistakes
        Ranking: 9
        CONSISTENCY: good till the end
        Ranking: 10
        STORY: no build up.
        Ranking:   <7>
        CREATIVITY: too generic
        Ranking:    6
        STYLE: fitting for a 5 year old.
        Ranking: 8

        Let me know if you'd like me to explain my reasoning for each category!
        """

    reply7 = """
    reply: Here are the ratings for each category:

    Grammar: The story has only one sentence, which is grammatically correct and simple in structure.

    SPELLING: Since there is no spelling to rate, I'll leave this one blank.

    CONSISTENCY: The story's consistency lies in its simplicity and brevity; it sticks to a straightforward narrative with no twists or turns.

    STORY: The story's plot is minimal, focusing on the concept of nothingness. While it may not be engaging for older readers, it could spark curiosity in young children about what "nothing" means.

    CREATIVITY: The idea of exploring the concept of "nothing" is intriguing and could lead to interesting discussions with children.

    STYLE: The story's tone is straightforward and matter-of-fact, which suits its purpose as a simple tale for young readers.

    Here are my ratings:

    GRAMMAR: 10
    SPELLING: N/A
    CONSISTENCY: 9
    STORY: 7
    CREATIVITY: 8
    STYLE: 8"""

    reply8 = """
    reply: Here are my ratings:

    Grammar: The story uses simple sentence structures and basic verb tenses, making it easy to follow for a 5-year-old reader.
    SPELLING: 
    CONSISTENCY: The story's tone, pace, and themes are consistent throughout, making it an enjoyable read.
    STORY:
    """

    reply9 = """
    reply: Here are the sentences for each category:

    * Grammar: The story's sentence structure is simple and easy to follow, but it could benefit from more varied sentence lengths.
    * Spelling: There is no specific spelling errors mentioned in this text.
    * Consistency: The story consistently uses a childlike tone and maintains its focus on Tom's adventure with Sam.
    * Story: The narrative follows a clear sequence of events and provides a relatable storyline for young readers.
    * Creativity: While the story has some creative elements, such as the monster game, it relies heavily on familiar tropes and doesn't take many risks.

    And here are my ratings:

    GRAMMAR: 7
    SPELLING: N/A (no spelling errors mentioned)
    CONSISTENCY: 8
    STORY: 6
    CREATIVITY: 5
    """

    reply10 = """
    reply: Here are the ratings:

    Grammar sentence: The story's grammatical structure is simplistic and lacks variety.

    GRAMMAR: 2
    SPELLING: N/A (since there's no spelling to rate)
    CONSISTENCY: 10 (the story consistently presents a simple, one-sentence plot)
    STORY: 1 (the story is extremely brief and lacks development or interest)
    CREATIVITY: 1 (the idea of telling a story about "nothing" is not particularly creative)
    STYLE: 2 (the writing style is straightforward but lacks flair or engaging language)

    Let me know if you'd like me to elaborate on any of these ratings!
    Error parsing prompt index 0: Failed to parse ratings for prompt index 0
    """

    reply11 = """
    Grammar: The story's grammar is simplistic and lacks proper sentence structure, but it still conveys a message.

    GRAMMAR: 2
    SPELLING: <RATING>

    Spelling: The story contains several misspellings, such as "nothink" instead of "nothing", which affect its readability.

    SPELLING: 4

    Consistency: The story's tone and style are consistent throughout, with a focus on simplicity and short sentences.

    CONSISTENCY: 8
    SPELLING:

    Story: The story is very simple and lacks a clear plot or character development, but it still attempts to convey an idea about the beginning of something.

    STORY: 5
    CREATIVITY:

    Creativity: While the story's concept is original, its execution is not particularly creative or engaging.

    CREATIVITY: 6
    STYLE:

    Style: The story's writing style is very simple and lacks descriptive language, but it still conveys a message in a straightforward manner.

    STYLE: 7

    """
    reply12 = """
    Here are my ratings:

    **Grammar:** The story's grammar is generally good, with only a few minor errors that do not affect the overall understanding of the text. Sentence structure is simple and easy to follow.

    Rating: 8

    **Spelling:** The spelling in the story appears to be accurate, with no notable errors or misspellings.

    Rating: 10

    **Consistency:** The story has a clear and consistent tone, with all events and actions making sense within the narrative. The characters' motivations and behaviors are also consistent throughout.
    
    Rating: 9
    
    **Story:** The story is sweet and heartwarming, with a clear moral about friendship and helping others. The plot is simple yet engaging for young readers.
    
    Rating: 7
    
    **Creativity:** While the idea of using a string to transmit music to a deaf cat is creative, the overall concept is not particularly original or surprising.
    
    Rating: 6
    
    **Style:** The language used in the story is clear and concise, making it easy for young readers to understand. The tone is also generally upbeat and cheerful.
    
    Rating: 8
    
    Here are my ratings in the required format:
    
    GRAMMAR: 8
    SPELLING: 10
    CONSISTENCY: 9
    STORY: 7
    CREATIVITY: 6
    STYLE: 8
    """

    categories = {
        0: "GRAMMAR",
        1: "SPELLING",
        2: "CONSISTENCY",
        3: "STORY",
        4: "CREATIVITY",
        5: "STYLE"
    }

    ratings1 = parse_prompt_3(reply1)
    ratings2 = parse_prompt_3(reply2)
    ratings3 = parse_prompt_3(reply3)
    ratings4 = parse_prompt_3(reply4)
    ratings5 = parse_prompt_3(reply5)
    ratings6 = parse_prompt_3(reply6)
    ratings7 = parse_prompt_3(reply7)
    ratings8 = parse_prompt_3(reply8)
    ratings9 = parse_prompt_3(reply9)
    ratings10 = parse_prompt_3(reply10)
    ratings11 = parse_prompt_3(reply11)
    ratings12 = parse_prompt_3((reply12))

    print("parse_prompt_3:\n")
    print(ratings1)
    print(ratings2)
    print(ratings3)
    print(ratings4)
    print(ratings5)
    print(ratings6)
    print(ratings7)
    print(ratings8)
    print(ratings9)
    print(ratings10)
    print(ratings11)
    print(ratings12)