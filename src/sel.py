import os
import time

import undetected_chromedriver as uc
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def main():
    load_dotenv()
    url = "https://seekingalpha.com/alpha-picks/subscribe"

    options = uc.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for 'LOG IN' button to appear before clicking
    login_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@data-test-id='header-button-sign-in']")
        )
    )
    login_button.click()

    # Wait for login form to appear by looking for 'Sign in' button
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@data-test-id='sign-in-button']")
        )
    )

    # Fill in email and password
    driver.find_element(By.XPATH, "//input[@name='email']").send_keys(
        os.getenv("ALPHA_EMAIL")
    )
    time.sleep(1)

    driver.find_element(By.XPATH, "//input[@name='password']").send_keys(
        os.getenv("ALPHA_PASSWORD")
    )
    time.sleep(1)

    signin_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@data-test-id='sign-in-button']")
        )
    )
    driver.execute_script("arguments[0].click();", signin_button)

    WebDriverWait(driver, 10).until(
        lambda driver: driver.current_url != "your_login_url"
    )

    time.sleep(20)
    driver.quit()


if __name__ == "__main__":
    main()
