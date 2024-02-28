
import os.path
from time import sleep
from random import randint

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support import expected_conditions as EC

# ------------------------------------------------------------------------------------------

def select_random_video(results_elements, number_of_related_videos_to_consider):
    return randint(0, min(len(results_elements), number_of_related_videos_to_consider))

# ------------------------------------------------------------------------------------------

def check_link(elem_text, elem_href):
    if elem_text == None or elem_href == None:
        return False
    if len(elem_text) <= 8 or len(elem_href) == 0:
        return False
    if "youtube" in elem_href:
        return True
    else:
        return False

# ------------------------------------------------------------------------------------------

def click_show_more(driver):

    try: 
        # First remove the snippet description and hashtags since it may contain links which make it impossible to click "show more"
        element = driver.find_element(By.ID, 'attributed-snippet-text')
        driver.execute_script("arguments[0].remove();", element)

        element = driver.find_element(By.ID, 'info-container')
        driver.execute_script("arguments[0].remove();", element)

        # Now click the element
        element = driver.find_element(By.ID, 'bottom-row')
        element.click()
    except Exception as e:
        print(e)

# ------------------------------------------------------------------------------------------

def click_show_transcript(driver, timeout=10):
    try:
        button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(.,"Transkript anzeigen")]'))
        )
        button.click()
        return 1
    except TimeoutException:
        print("Timeout: Transcript button not found or could not be clicked within {} seconds.".format(timeout))
        return 0
    except Exception as e:
        print("An error occurred:", str(e))
        return 0

# ------------------------------------------------------------------------------------------

def click_element_of_class(driver, class_name):
    try:
        # class_name = "style-scope ytd-text-inline-expander"
        elements = driver.find_elements(By.CLASS_NAME, value=class_name)

        try:
            for element in elements:
                element.click()
        except:
            pass

    except Exception as e:
        print(f"'{class_name}' not found or could not be clicked:", str(e))

# ------------------------------------------------------------------------------------------

def accept_cookies(driver):
    click_button(driver, button_text="Alle akzeptieren")

# ------------------------------------------------------------------------------------------

def click_show_transcript(driver):
    click_button(driver, button_text="Transkript anzeigen")

# ------------------------------------------------------------------------------------------

def click_button(driver, button_text, timeout=20):
    try:
        button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, f'//button[contains(.,"{button_text}")]'))
        )
        button.click()
    except TimeoutException:
        print(f"Timeout: {button_text} button not found or could not be clicked within {timeout} seconds.")
    except Exception as e:
        print("An error occurred:", str(e))

# ------------------------------------------------------------------------------------------

def scroll_down(driver, scroll_pause_time=2, iterations=2):

    html = driver.find_element(By.TAG_NAME, 'html')

    for _ in range(iterations):
        html.send_keys(Keys.END)
        time.sleep(scroll_pause_time)

# ------------------------------------------------------------------------------------------

def crawl_youtube(keywords, number_of_paths_to_collect_per_keyword,
                  number_of_related_videos_to_consider,
                  number_of_related_videos_to_visit_depth, base_path, folder_prefix,
                  browser="firefox"):

    if browser == "firefox":
        driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    else:
        print(f"Browser {browser} not supported.")
        return

    search_item_delimiter = " END_OF_SEARCH_ITEM "

    for qidx, query in tqdm(enumerate(keywords)):
        print("Queries:", qidx + 1, "/", len(keywords))

        query_ = query.replace(' ', '_').replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")
        folder_prefix_ = os.path.join(folder_prefix, f"{qidx}_" + query_)

        folder_path = os.path.join(base_path, folder_prefix_)

        if os.path.isdir(folder_path):
            print("  already done")
            continue
        else:
            print(">>>", folder_path)

        for i in range(number_of_paths_to_collect_per_keyword):
            print("  Paths per keyword:", i+1, "/", number_of_paths_to_collect_per_keyword)

            # driver.delete_all_cookies()

            try:
                filepath_base = os.path.join(base_path, folder_prefix_)

                if not os.path.exists(filepath_base):
                    os.makedirs(filepath_base)

                search_term = query.replace(" ","%20")

                current_url = "https://youtube.de/results?search_query=" + search_term

                # ------------------------
                print(query.encode('utf-8'), "---", i, "---", current_url.encode('utf-8'))

                #print("Open URL")
                driver.get(current_url)

                #print("Accept Cookies")
                accept_cookies(driver)

                # ------------------------

                # wait until YouTube is fully loaded
                sleep(20)

                filename = current_url.replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")

                len_fn = len("results_search_query_") + len(query)
                filename = filename[-len_fn:] + ".html"
                filepath = os.path.join(filepath_base, filename)

                # collect top recommendations
                results_elements = [[elem.text, elem.get_attribute("href")] for elem in driver.find_elements(By.ID, "video-title") if elem.text and elem.get_attribute("href")]

                # --------------------

                for _ in range(0, 100):

                    selected_element_i = select_random_video(results_elements, number_of_related_videos_to_consider)
                    print(len(results_elements), selected_element_i)

                    selected_element = results_elements[selected_element_i]
                    elem_text = selected_element[0]
                    elem_href = selected_element[1]

                    if not "/shorts/" in elem_href:
                        break

                visited_pages_and_filenames = [query, str(i), str(selected_element_i), str(len(results_elements)),
                                               elem_text, elem_href, filepath, search_item_delimiter]

                # follow related videos
                for j in range(number_of_related_videos_to_visit_depth):
                    print("    depth:", j+1, "/", number_of_related_videos_to_visit_depth)
                    driver.get(elem_href)

                    path_per_depth = os.path.join(filepath_base, f"Depth {j}")

                    if not os.path.exists(path_per_depth):
                        os.makedirs(path_per_depth)

                    if "/shorts/" in elem_href:
                        continue

                    print(query.encode('utf-8'), "---", i, "---", j, "---", elem_href.encode('utf-8'))

                    # wait until YouTube is fully loaded
                    myElem = WebDriverWait(driver, 10000).until(EC.presence_of_element_located((By.ID, 'comments')))

                    print("click 'show more'")
                    click_show_more(driver)

                    print("click 'show transcript")
                    status = click_show_transcript(driver)

                    if status == 0:
                        j -= 1

                    time.sleep(2)

                    print("scrolling down")
                    scroll_down(driver, scroll_pause_time=2, iterations=2)

                    filename = elem_href.replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")
                    filename = filename + ".html"
                    filepath = os.path.join(path_per_depth, filename)

                    # save source code of website
                    if not os.path.isfile(filepath):
                        try:
                            with open(filepath, 'w', encoding='utf8') as f:
                                f.write(driver.page_source)
                        except Exception as e:
                            print(e)

                    related_elements = driver.find_elements(By.ID, "related")[0]

                    related_elements_links = [[elem.text, elem.get_attribute("href")] for elem in related_elements.find_elements(By.CLASS_NAME, "yt-simple-endpoint")
                                               if check_link( elem.text, elem.get_attribute("href"))]
                    try:
                        selected_element_i = select_random_video(related_elements_links, number_of_related_videos_to_consider)
                        selected_element = related_elements_links[ selected_element_i ]
                    except Exception as e:
                        print(e)
                        j -= 1
                        continue

                    elem_text = selected_element[0]
                    elem_href = selected_element[1]

                    visited_pages_and_filenames.extend((str(j+1), str(selected_element_i), str(len(related_elements_links)), elem_text, elem_href, filepath, search_item_delimiter))

                crawling_paths_csv_fn = os.path.join(filepath_base, "crawling_paths.csv")

                # save file that contains all the video URLs and how they relate
                with open(crawling_paths_csv_fn, "a", encoding='utf8') as f:
                    f.write("\t".join(visited_pages_and_filenames).replace("\n","") + "\n")

            except Exception as e:
                raise(e)
