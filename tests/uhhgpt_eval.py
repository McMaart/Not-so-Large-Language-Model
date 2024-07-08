"""
Script for the automatic evaluation of the stories generated from our models with UHHGPT.
For this script to work, the user must fulfill the following requirements:
1. The user must have the Chrome browser installed and download a suitable version of the Chromedriver
(see https://googlechromelabs.github.io/chrome-for-testing/). The chromedriver.exe file must be located in the PATH (or
alternatively in this folder).
2. The user must create an environment variable with the variable name "PASS_UNI". The value of this variable is
the (seven-digit) name of the user account for the UHH (e.g. 'BAO1234'), followed by the corresponding password.
Example of the value of the environment variable (for the user account 'BAO1234' with the password 'password420'):
BAO1234Password420
"""
import sys
import time
from os import environ
import numpy as np
from selenium.common import NoSuchElementException
from selenium.webdriver import Keys, Chrome, Firefox
from selenium.webdriver.common.by import By
from prompt_testing.parse_model_reply import parse_prompt_0
np.set_printoptions(precision=4)


class ChromeDriver(Chrome):
    def __init__(self, wait: int = 10):
        super().__init__()
        self.maximize_window()
        self.implicitly_wait(wait)

    def click_css_element(self, selector: str):
        self.find_element(By.CSS_SELECTOR, selector).click()


def get_response(driver: ChromeDriver, wait: int | float = 3, max_iter: int = 100):
    prev_response = "NAN"
    for _ in range(max_iter):
        time.sleep(wait)

        cur_response = driver.find_element(By.CSS_SELECTOR, 'body > div > div.main > div.messages > div:nth-child(3) > '
                                                            'div > div.message-text').text
        if cur_response == prev_response:
            return cur_response
        prev_response = cur_response


def setup() -> ChromeDriver:
    # Login
    driver = ChromeDriver()
    driver.get("https://uhhgpt.uni-hamburg.de/login.php")
    driver.click_css_element('body > div > aside > div.loginPanel > form > button')  # login button on title page
    try:
        user_elem = driver.find_element(By.CSS_SELECTOR, '#username')
    except NoSuchElementException as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    user_elem.send_keys(environ.get('PASS_UNI')[:7])
    pass_elem = driver.find_element(By.CSS_SELECTOR, '#password')
    pass_elem.send_keys(environ.get('PASS_UNI')[7:])
    driver.click_css_element('#inhalt > main > section.spalten > article > div:nth-child(2) > div > div > '
                             'div.loginform > div.left > form > div.form-element-wrapper > button')  # confirm login
    driver.click_css_element('#data-protection > div > button')  # accept terms of use

    # Change to GPT-4o model
    # choose the GPT-4o-model from the menu
    driver.click_css_element('body > div > div.sidebar > div.menu > div.radiogroup > label:nth-child(2) > p')
    driver.click_css_element('body > div > div.main > div.input-container > center > button')  # start new chat
    return driver


def get_ratings(driver: ChromeDriver, instruction: str, generated_stories: list[str]) -> list[list[int | None]]:
    story_ratings = []
    for story in generated_stories:
        full_prompt = instruction.format(story)
        prompt_elem = driver.find_element(By.CSS_SELECTOR, '#texreachat')
        driver.execute_script("arguments[0].value = arguments[1];", prompt_elem, full_prompt)
        prompt_elem.send_keys(Keys.RETURN)

        answer = get_response(driver)
        story_ratings.append(parse_prompt_0(answer))

        driver.click_css_element('body > div > div.main > div.input-container > center > button')  # new chat
        print(f"UHHGPT:\n{answer}", end="\n\n")

    return story_ratings


if __name__ == '__main__':
    # ToDo: Import (/replace with) the best prompt
    sample_prompt = """
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
    sample_story = """
    Once upon a time, there was a little girl named Lily. She had a big, red ball that she loved to play with. One day, she saw a small, helpless bird on the ground. The bird could not fly. Lily wanted to help the bird. Lily said to the bird, "Don't worry, little bird. I will help you." She picked up the bird and took it to her mom. Her mom knew how to help the bird. Her mom said, "We can give the bird some food and water." Lily and her mom gave the bird some food and water. The bird started to feel better. It was not helpless anymore. The bird said, "Thank you, Lily and mom." Lily was happy to help the bird. She learned that helping others can make you feel good too. And from that day on, Lily always tried to help others when they needed it.
    """
    stories = [sample_story, sample_story, sample_story]

    driver = setup()
    ratings = get_ratings(driver, sample_prompt, stories)
    print(f"Rating list: {ratings}")

    rating_arr = np.array(ratings)
    print(f"Avg. rating: {rating_arr.mean(axis=0)}")
    print(f"Std:         {rating_arr.std(axis=0)}")
