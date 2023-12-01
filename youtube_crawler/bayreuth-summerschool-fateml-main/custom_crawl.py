# -*- coding: utf-8 -*-
import sys
import os.path
from time import sleep
from random import randint, shuffle

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager

#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support import expected_conditions as EC

# parameters
NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD = 20

# keywords which are used
keywords = [
  "kopp verlag",
  "covid wahrheit",
]

# randomize order of keywords
shuffle( keywords )

# main loop
for keyword in keywords:
  for i in range( NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD ):

    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    driver.delete_all_cookies()

    try:
      search_term = keyword.replace(" ","%20")

      current_url = "https://www.amazon.de/s?k=" + search_term

      print( keyword.encode('utf-8'), "---", i, "---", current_url.encode('utf-8') )

      # open url
      driver.get( current_url )

      # wait until YouTube is fully loaded
      sleep(3)

      # prepare filname to save file
      filename = current_url.replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")
      filename = filename[-31:]
      filepath = 'crawled_pages/amazon_' + filename + ".html" 

      if not os.path.isfile( filepath ):
        with open( filepath, 'w', encoding='utf8' ) as f:
          f.write( driver.page_source )

      for elem1 in driver.find_elements_by_id("search"):
        for elem2 in elem1.find_elements_by_tag_name("h2"):
          print( elem2.text )
          for elem3 in elem2.find_elements_by_tag_name("a"):
            print( elem3.get_attribute("href") )

          print()
    finally:
      driver.close()