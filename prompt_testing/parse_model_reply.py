'''
For each prompt listed in "prompts.txt", this script includes a corresponding function that parses the model's reply to that prompt.
Each function should return a list with 6 numbers from 1 to 10 for each of the 6 categories:
0 - GRAMMAR
1 - SPELLING
2 - CONSISTENCY
3 - STORY
4 - CREATIVITY
5 - STYLE
'''
categories = {
    0: "GRAMMAR",
    1: "SPELLING",
    2: "CONSISTENCY",
    3: "STORY",
    4: "CREATIVITY",
    5: "STYLE"
}

def parse_model_reply(reply: str, prompt_idx: int) -> list[int] | None:
    return parse_functions.get(prompt_idx)(reply)

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

parse_functions = {
    0: parse_prompt_0,
    1: parse_prompt_0
}