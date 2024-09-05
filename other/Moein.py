from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.by import By

import time
import requests

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url_cite = "https://ticketleader.evenue.net/cgi-bin/ncommerce3/SEGetEventInfo?ticketCode=GS%3ATKTLDR%3ACOL24%3A1116MO%3A&linkID=tktldr&shopperContext=&pc=&caller=&appCode=&groupCode=MOEIN&cgc=&dataAccId=335&locale=en_US&siteId=ev_tktldr"

TOKEN= '6813501660:AAFFN6WlupQbZc_NrLzTYdVuR-LEJNNmtgs'
chat_ids = ['73859597']
for chat_id in chat_ids:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text=Bot is running looking for bunzen lake availability date. " 
    print(requests.get(url).json())
while True:
  try:
      driver.get(url_cite)
      time.sleep(10)
      if True:
        today = driver.find_element(By.ID, 'U:33')
        print(today)
        exit(0)

        if today.fill != 'SOLD OUT':
            for chat_id in chat_ids:
              url1 = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text=spot available! hurry up!"
              print(requests.get(url1).json())
      
      elif Day == "Tomorrow":

        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "smartSelectCustom"))  # Replace with the actual class name
        )
        button.click()
        time.sleep(5)
        element2 = driver.find_element(By.CSS_SELECTOR,
                                      'div.popup.smart-select-popup.modal-in > div.view > div.page.smart-select-page.page-with-subnavbar > div.page-content > div > ul > li:nth-child(2)')

        element2.click()
        time.sleep(5)
        element3 = driver.find_element(By.CSS_SELECTOR,
                                      'div.sub-padder > div.screenCenter.paymentMain > div:nth-child({div_selector}) > div.cardDetails > div.card-header > div.accordion-item-opened.accordion-item > div.mainHeading.accordion-item-toggle > div:nth-child(2) > div:nth-child(1)')

        if element3.text != 'SOLD OUT':
            for chat_id in chat_ids:
              url1 = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text=spot available! hurry up!"
              print(requests.get(url1).json())

          


  # except:
  #     for chat in chat_ids:
  #       url_error = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text=Error in the script"
  #       print(requests.get(url_error).json())
  finally:
      pass
      # Close the browser
driver.quit()