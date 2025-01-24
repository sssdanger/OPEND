import requests
import os
from tqdm import tqdm
import pickle
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import random
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from lxml import etree


def get_options():
    options = webdriver.ChromeOptions()
    
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) 
    options.add_experimental_option('useAutomationExtension', False)


    service = Service('chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined
        })
      """
    })
    
    return driver


driver = get_options()
def extract_news_and_labels(file_path):
    news_contents = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:

            line = line.strip()

            parts = line.split('\t')
            if len(parts) == 2:
                label, news_content = parts
                news_contents.append(news_content)
                labels.append(label)
            else:
                print(f"格式错误：{line}")

    return news_contents, labels

def search_google_search_api(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

class AbnormalTrafficDetectedException(Exception):
    pass

def check_for_abnormal_traffic(driver):
    try:

        if "异常流量" in driver.page_source:
            raise AbnormalTrafficDetectedException("Abnormal traffic detected! Stopping the script.")
    except NoSuchElementException:
        pass  

def search_google_search_selenium(query):
    driver.get('https://www.google.com')

    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)  


    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(random.uniform(5, 15))
    check_for_abnormal_traffic(driver)
    html = driver.page_source
    if len(html) > 0:
        print("成功获取到 HTML 内容")
    else:
        print("未获取到 HTML 内容")
    soup = BeautifulSoup(html, 'html.parser')
    link = []
    jsname_a_tags = soup.find_all('a', attrs={'jsname': 'UWckNb'})
    
    for a_tag in jsname_a_tags:
        link.append(a_tag.get('href'))

    return link

def save_results_to_tsv(query, results, tsv_file_path):
    with open(tsv_file_path, 'a', encoding='utf-8') as tsv_file:  
        for item in results.get('items', []):
            tsv_file.write(f"{query}\t{item.get('title', '')}\t{item.get('snippet', '')}\t{item.get('link', '')}\n")


def save_results_to_tsv_sele(query, results, tsv_file_path):
    with open(tsv_file_path, 'a', encoding='utf-8') as tsv_file:  
        for item in results:
            tsv_file.write(f"{query}\t{item}\n")

api_key = ''
engine_id = ''





file_path = ''
news_contents, labels = extract_news_and_labels(file_path)
query_list = news_contents
tsv_file_path = '' 
unique_claim_texts = set()
with open(tsv_file_path, "r", encoding="utf-8") as file:
    for line in file:

        fields = line.strip().split("\t")
        

        if len(fields) == 2:
            claim_text = fields[0]  
            unique_claim_texts.add(claim_text)

num = 0
for query in tqdm(query_list, desc="Processing Query"):
    if query in unique_claim_texts:
        continue
    num+=1
    if num>80 :
        num = 0
        driver.quit()
        # time.sleep(60*10)
        driver = get_options()
    results = search_google_search_selenium(query)
    if results:
        save_results_to_tsv_sele(query, results, tsv_file_path)


