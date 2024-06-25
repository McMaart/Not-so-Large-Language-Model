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
from selenium.common import NoSuchElementException
from selenium.webdriver import Chrome, Keys
from selenium.webdriver.common.by import By

uhhgpt_website = {'login_button': '#inhalt > main > section.spalten > article > div:nth-child(2) > div > div > '
                                  'div.loginform > div.left > form > div.form-element-wrapper > button',
                  'pass': '#password',
                  'user': '#username'}


class ChromeDriver(Chrome):
    def __init__(self, wait: int = 10):
        super().__init__()
        self.maximize_window()
        self.implicitly_wait(wait)

    def click_css_element(self, selector):
        elem = self.find_element(By.CSS_SELECTOR, selector)
        elem.click()


def get_response(driver: ChromeDriver, wait=3, max_iter=100):
    prev_response = "NAN"
    for _ in range(max_iter):
        time.sleep(wait)

        cur_response = driver.find_element(By.CSS_SELECTOR, 'body > div > div.main > div.messages > div:nth-child(3) > '
                                                            'div > div.message-text').text
        if cur_response == prev_response:
            return cur_response
        prev_response = cur_response


# Login
driver = ChromeDriver()
driver.get("https://uhhgpt.uni-hamburg.de/login.php")
driver.click_css_element('body > div > aside > div.loginPanel > form > button')
try:
    user_elem = driver.find_element(By.CSS_SELECTOR, '#username')
except NoSuchElementException as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)

user_elem.send_keys(environ.get('PASS_UNI')[:7])
pass_elem = driver.find_element(By.CSS_SELECTOR, uhhgpt_website.get('pass'))
pass_elem.send_keys(environ.get('PASS_UNI')[7:])
driver.find_element(By.CSS_SELECTOR, uhhgpt_website.get('login_button')).click()

# Change to GPT-4o
driver.click_css_element('#data-protection > div > button')
driver.click_css_element('body > div > div.sidebar > div.menu > div.radiogroup > label:nth-child(2) > p')
new_chat = driver.find_element(By.CSS_SELECTOR, 'body > div > div.main > div.input-container > center > button')
new_chat.click()

# Prompt to the model
sample_prompt = "Hello, is the university of Hamburg (UHH) excellent?"
prompt_elem = driver.find_element(By.CSS_SELECTOR, '#texreachat')
prompt_elem.send_keys(f"{sample_prompt}?{Keys.RETURN}")
answer = get_response(driver)
print(f"Chat_GPT: {answer}")
time.sleep(1200)
