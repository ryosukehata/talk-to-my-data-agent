# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By

from tests.e2e.utils import (
    click_element,
    download_file,
    find_element,
    wait_for_element_to_be_visible,
)


def assert_data_processed(browser) -> None:
    assert wait_for_element_to_be_visible(
        browser,
        By.XPATH,
        "//p[contains(text(), '✅ Data processed and dictionaries generated successfully!')]",
        120,
    )


def load_from_database(browser: webdriver.Chrome, dataset: str) -> None:
    click_element(
        browser, By.CSS_SELECTOR, 'div[data-testid="stExpander"]:nth-of-type(3)'
    )

    assert wait_for_element_to_be_visible(
        browser,
        By.XPATH,
        "//p[contains(text(), 'Load Datasets from Snowflake')]",
    )

    click_element(
        browser,
        By.XPATH,
        "(//div[contains(text(), 'Choose an option')])[2]",
    )

    click_element(
        browser,
        By.XPATH,
        f"//div[contains(text(), '{dataset}')]",
    )

    # click on another element to close the dropdown
    click_element(
        browser,
        By.XPATH,
        "//p[contains(text(), 'Select datasets from AI Catalog')]",
    )

    click_element(
        browser,
        By.XPATH,
        "//p[contains(text(), 'Load Selected Tables')]",
    )

    assert_data_processed(browser)


@pytest.mark.usefixtures("check_if_logged_in")
def test_app_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)
    assert wait_for_element_to_be_visible(
        browser,
        By.XPATH,
        "//p[contains(text(), 'Upload and process your data using the sidebar to get started')]",
        60,
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_load_from_database(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)
    load_from_database(browser, "LENDING_CLUB_PROFILE")

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'View Cleaning Report')]", 60
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_data_dictionary_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)
    load_from_database(browser, "LENDING_CLUB_PROFILE")

    click_element(
        browser,
        By.XPATH,
        "//span[contains(text(), 'Data Dictionary')]",
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Download Data Dictionary')]"
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_chat_page_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)
    load_from_database(browser, "LENDING_CLUB_PROFILE")

    click_element(
        browser,
        By.XPATH,
        "//span[contains(text(), 'Chat With Data')]",
    )

    assert wait_for_element_to_be_visible(
        browser, By.CSS_SELECTOR, 'textarea[data-testid="stChatInputTextArea"]'
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_file_upload(browser: webdriver.Chrome, get_app_url: str) -> None:
    ten_k_diabetes_20 = download_file(
        browser,
        "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv",
    )
    browser.get(get_app_url)

    file_input = find_element(
        browser, By.CSS_SELECTOR, 'input[data-testid="stFileUploaderDropzoneInput"]'
    )
    file_input.send_keys(str(ten_k_diabetes_20))

    assert_data_processed(browser)

    click_element(
        browser,
        By.XPATH,
        "//span[contains(text(), 'Data Dictionary')]",
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Download Data Dictionary')]"
    )

    click_element(
        browser,
        By.XPATH,
        "//span[contains(text(), 'Chat With Data')]",
    )

    chat_input = wait_for_element_to_be_visible(
        browser, By.CSS_SELECTOR, 'textarea[data-testid="stChatInputTextArea"]'
    )

    chat_input.send_keys("What is the most common medical specialty of the physician?")

    click_element(
        browser, By.CSS_SELECTOR, 'button[data-testid="stChatInputSubmitButton"]'
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Bottom Line')]", 60
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Analysis Code')]", 60
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Analysis Results')]", 60
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Additional Insights')]", 60
    )

    assert wait_for_element_to_be_visible(
        browser, By.XPATH, "//p[contains(text(), 'Follow-up Questions')]", 60
    )
