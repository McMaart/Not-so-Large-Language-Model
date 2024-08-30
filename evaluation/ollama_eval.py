"""
Function for prompting to the LLama model.
"""
import sys
import requests
import json

url = "http://localhost:11434/api/generate"
headers = {'Content-Type': 'application/json'}


def get_llama_response(prompt: str) -> str | None:
    request_data = {
        "model": "llama3",
        "stream": False,
        "prompt": prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(request_data))
    if response.status_code == 200:
        data = json.loads(response.text)
        return data["response"]
    else:
        print(f"Error: {response.status_code}, {response.text}", file=sys.stderr)
        return None


if __name__ == "__main__":
    from prompts_gpt_eval import prompt_v2

    sample_story = ("Once upon a time there was a little girl named Lucy. She was three years old and lived in a big "
                    "house with her mom and dad. Lucy's room was very messy. She saw a box and wanted to remove it, "
                    "so her dad had to remove it from the closet. So lucy asked her dad for help. Her dad carefully "
                    "removed the box from the closet and put it somewhere safe. It was heading around the house, "
                    "keeping things neat and tidy after sorting them all. Lucy was proud of her work. Then her dad "
                    "said he was very proud of her. Lucy was very happy to have spot's help and she was happy that "
                    "she was able to remove the box easily. She had to keep it tidy and tidy. From then on, "
                    "she always kept her room tidy. And she always put her toys away in the closet when she got out. "
                    "The end!")
    print(get_llama_response(prompt_v2.format(sample_story)))
