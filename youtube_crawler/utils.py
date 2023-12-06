# -*- coding: utf-8 -*-
import sys
import os.path
from time import sleep
from random import randint, shuffle

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# ------------------------------------------------------------------------------------------
def select_random_video(results_elements, number_of_related_videos_to_consider):
    return randint(0, min(len(results_elements), number_of_related_videos_to_consider))

# ------------------------------------------------------------------------------------------
def check_link(elem_text, elem_href):
    if elem_text == None or elem_href == None:
        return False
    if len( elem_text ) <= 8 or len( elem_href ) == 0:
        return False
    if "youtube" in elem_href:
        return True
    else:
        return False

# ------------------------------------------------------------------------------------------

def crawl_youtube(keywords, number_of_paths_to_collect_per_keyword,
                  number_of_search_results_to_consider,
                  number_of_related_videos_to_consider,
                  number_of_related_videos_to_visit_depth, base_path, folder_prefix="folder"):
    # main loop
    for keyword in keywords:
        for i in range(number_of_paths_to_collect_per_keyword):

            driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
            driver.delete_all_cookies()

            try:
                filepath_base = os.path.join(base_path, "crawled_pages", folder_prefix)

                if not os.path.exists(filepath_base):
                    os.makedirs(filepath_base)

                search_term = keyword.replace(" ","%20")

                current_url = "https://youtube.de/results?search_query=" + search_term

                print(keyword.encode('utf-8'), "---", i, "---", current_url.encode('utf-8'))

                # open url
                driver.get( current_url )

                # ------------------------

                # Accept cookies
                try:
                    cookies_button = WebDriverWait(driver, 10000).until(
                        EC.element_to_be_clickable((By.XPATH, '//button[contains(.,"Alle akzeptieren")]'))
                    )
                    cookies_button.click()
                except Exception as e:
                    print("Cookies button not found or could not be clicked:", str(e))

                # ------------------------

                # wait until YouTube is fully loaded
                myElem = WebDriverWait( driver, 10000 ).until( EC.presence_of_element_located((By.ID, 'comments')))

                # prepare filname to save file
                filename = current_url.replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")

                len_fn = len("results_search_query_") + len(keyword)
                filename = filename[-len_fn:]
                # filepath = 'crawled_pages/' + filename + ".html"
                filepath = os.path.join(filepath_base, filename + ".html")

                if not os.path.isfile( filepath ):
                    with open( filepath, 'w', encoding='utf8' ) as f:
                        f.write( driver.page_source )

                # collect top recommendations
                results_elements = [ [elem.text, elem.get_attribute("href")] for elem in driver.find_elements(By.ID, "video-title") if elem.text and elem.get_attribute("href") ]

                # --------------------

                selected_element_i = select_random_video(results_elements, number_of_related_videos_to_consider)
                selected_element = results_elements[ selected_element_i ]

                elem_text = selected_element[ 0 ]
                elem_href = selected_element[ 1 ]

                visited_pages_and_filenames = [ keyword, str(i), str(selected_element_i), str(len(results_elements)), elem_text, elem_href, filepath ]

                # follow related videos
                for j in range(number_of_related_videos_to_visit_depth):
                    driver.get( elem_href )

                    print( keyword.encode('utf-8'), "---", i, "---", j, "---", elem_href.encode('utf-8') )

                    # wait for a random amount of time
                    sleep( randint(500,7000) / 1000.0 )

                    # wait until YouTube is fully loaded
                    myElem = WebDriverWait( driver, 10000 ).until( EC.presence_of_element_located((By.ID, 'comments')))

                    filename = elem_href.replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")
                    filename = filename[-31:]
                    filename = filename + ".html"

                    filepath = os.path.join(filepath_base, filename + ".html")

                    # save source code of website
                    if not os.path.isfile( filepath ):
                        with open( filepath, 'w', encoding='utf8' ) as f:
                            f.write( driver.page_source )

                    related_elements = driver.find_elements(By.ID, "related")[0]

                    related_elements_links = [ [elem.text, elem.get_attribute("href")] for elem in related_elements.find_elements(By.CLASS_NAME, "yt-simple-endpoint")
                                               if check_link( elem.text, elem.get_attribute("href") ) ]

                    try:
                        selected_element_i = select_random_video(related_elements_links, number_of_related_videos_to_consider)
                        selected_element = related_elements_links[ selected_element_i ]
                    except Exception as e:
                        print(e)
                        j -= 1
                        continue

                    elem_text = selected_element[ 0 ]
                    elem_href = selected_element[ 1 ]

                    visited_pages_and_filenames.extend(( str(j+1), str(selected_element_i), str(len(related_elements_links)), elem_text, elem_href, filepath ))

                # save file that contains all the video URLs and how they relate
                with open('crawling_paths.csv', "a", encoding='utf8') as f:
                    print( "\t".join( visited_pages_and_filenames ).replace("\n","").encode('utf-8') )

                    f.write( "\t".join( visited_pages_and_filenames ).replace("\n","") + "\n" )
            except Exception as e:
                raise( e )
                pass

            driver.close()