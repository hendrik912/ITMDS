import torch
import gc
import re
import joblib
import string
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate.utils import release_memory
from langdetect import detect
from tqdm import tqdm_notebook
from collections import Counter
from bs4 import BeautifulSoup
from glob import glob

# -------------------------------------------------------------------------

def calculate_entropy_normalized(categories):
    """
    Calculate the entropy of a list of categorical values.
    
    Parameters:
        categories (list): List of categorical values.
        
    Returns:
        float: Entropy value normalized based on the maximum possible entropy.
    """
    # Count the occurrences of each category
    category_counts = np.unique(categories, return_counts=True)[1]
    
    # Compute probabilities
    probabilities = category_counts / len(categories)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Compute maximum possible entropy based on number of categories
    max_entropy = np.log2(len(probabilities))
    
    # Normalize entropy based on maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

# -----------------------------------------------------------------------

def load_data(filename, personas, persona_queries, data_path, overwrite=False, print_data=False, max_depth=5):

    if overwrite: 
        compute_new = True
    else:
        try:
            data_dict = joblib.load(filename)
            return data_dict
        except:
            compute_new = True

    data_dict = None

    if compute_new: 
        data_dict = {}

        for persona in personas:
            print(persona)

            data_dict[persona] = {}

            queries = persona_queries[persona]

            for qidx, query in enumerate(queries): 

                data_dict[persona][qidx] = {}

                query_ = query.replace(' ', '_').replace(":","_").replace("/","_").replace(".","_").replace("?","_").replace("=","_")
                query_ = f"{qidx}_{query_}"

                for depth in range(0, max_depth):
                    
                    data_dict[persona][qidx][depth] = {}

                    persona_data_path = os.path.join(data_path, persona, query_, f"Depth {depth}")

                    print(persona_data_path)

                    for idx, fname in enumerate(glob(persona_data_path + "/*")):

                        if "crawling_paths.csv" in fname:
                            continue

                        data_dict[persona][qidx][depth][idx] = {}
                        data_dict[persona][qidx][depth][idx]["file_name"] = fname

                        soup = BeautifulSoup(open(fname, 'r', encoding='utf-8').read(), 'html.parser' )

                        # ----------------------
                        # Channel Name

                        list = soup.select(".style-scope ytd-channel-name")

                        if len(list) > 0:
                            channel_name = clean_elem(list[0].getText())

                            # Remove 'bestätigt' and strip any leading/trailing whitespace
                            cleaned_string = channel_name.replace('bestätigt', '').strip()

                            # Split the string using four consecutive spaces as delimiter
                            channels = cleaned_string.split("    ")  # Four spaces within the quotes

                            # Get the first channel name
                            channel_name = channels[0]

                            if print_data:
                                print("Channel name:", channel_name)

                            data_dict[persona][qidx][depth][idx]["channel_name"] = channel_name
                        else:
                            data_dict[persona][qidx][depth][idx]["channel_name"] = "not found"

                        # ----------------------
                        # Video Title

                        try:
                            parent = soup.find('div', {'id': 'title', 'class': 'style-scope ytd-watch-metadata'})
                            child = parent.find('yt-formatted-string', class_='style-scope ytd-watch-metadata')
                            if print_data:
                                print("Video title:", child.getText())

                            data_dict[persona][qidx][depth][idx]["video_title"] = child.getText()

                        except Exception as e:
                            data_dict[persona][qidx][depth][idx]["video_title"] = "not found"
                            print(e)

                        # # ----------------------
                        # Views

                        list = soup.select("#info-text .view-count")

                        if len(list) > 0:
                            views = clean_elem(list[0].getText())
                            data_dict[persona][qidx][depth][idx]["views"] = views
                            if print_data:
                                print("Views:", views)
                        else:
                            data_dict[persona][qidx][depth][idx]["views"] = "not found"

                        # ----------------------
                        # Subscribers

                        list = soup.select("#owner-sub-count")

                        if len(list) > 0:
                            channel_subscribers = clean_elem(list[0].getText())
                            if print_data:
                                print("Channel subscribers:", channel_subscribers)
                            data_dict[persona][qidx][depth][idx]["subscribers"] = channel_subscribers
                        else:
                            data_dict[persona][qidx][depth][idx]["subscribers"] = "not found"

                        # ----------------------
                        # Likes

                        try:
                            parent = soup.find('div', {'class': 'YtSegmentedLikeDislikeButtonViewModelSegmentedButtonsWrapper'})
                            child = parent.find('div', class_='yt-spec-button-shape-next__button-text-content')
                            if print_data:
                                print("Likes:", child.getText())
                            data_dict[persona][qidx][depth][idx]["likes"] = child.getText()
                        except Exception as e:
                            print(e)
                            data_dict[persona][qidx][depth][idx]["likes"] = "not found"

                        # ----------------------
                        # Comments

                        list = soup.select("#content-text")

                        if len(list) > 0:
                            comments = []

                            for comment_idx in range(len(list)):
                                comments.append(clean_elem(list[comment_idx].getText()))

                            if print_data:
                                print("Number of comments:", len(comments))
                        
                            data_dict[persona][qidx][depth][idx]["comments"] = comments
                        else:
                            data_dict[persona][qidx][depth][idx]["comments"] = []

                        # ----------------------
                        # Transcript

                        list = soup.find_all('yt-formatted-string', class_='segment-text style-scope ytd-transcript-segment-renderer')

                        if len(list) > 0:
                            sentences = []

                            for sentence_idx in range(len(list)):
                                sentences.append(clean_elem(list[sentence_idx].getText()))

                            if len(sentences) > 0:
                                transcript = " ".join(sentences)
                            else:
                                transcript = ""

                            data_dict[persona][qidx][depth][idx]["transcript"] = transcript
                        else:
                            data_dict[persona][qidx][depth][idx]["transcript"] = []

        joblib.dump(data_dict, filename)
        return data_dict

# -----------------------------------------------------------------------

def extract_pattern(text, pattern=r'\{.*\}'):

    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None  
    
# -----------------------------------------------------------------------

def clean_elem(elem):
    return elem.replace("\n", " ").strip()

# -----------------------------------------------------------------------

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# -----------------------------------------------------------------------

def preprocess_text(text, lowercasing=True, remove_punctuation=True, remove_stop_words=True, lemmatization=True, verbose=0):
    
    if lowercasing:
        text = text.lower()

    if remove_punctuation:    
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    if remove_stop_words:
        # Detect the language
        try:
            language = detect(text)
        except:
            language = 'default'

        if language == 'en':
            stop_words = set(stopwords.words("english"))
        elif language == 'de':
            stop_words = set(stopwords.words("german"))
        else:
            stop_words = set(stopwords.words("english"))
            if verbose>0:
                if len(text) > 200: 
                    text_sample = text[:200]
                else:
                    text_sample = text

                print("Language defaulted to english:", text_sample)

        words = text.split()
        words = [word for word in words if word not in stop_words]
    else:
        words = text.split()
        
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
    # Join the words back into a single string
    processed_text = " ".join(words)
    
    return processed_text

# -----------------------------------------------------------------------

def load_pipeline(use="mistral", token=""):

    if use=="tiny":
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        pipe = pipeline(
            "text-generation", 
            model=model_name, 
            device_map="auto", 
            return_full_text=False,
        )
        return pipe
    
    elif use == "falcon":

        model_name = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_4bit=True,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            tokenizer=tokenizer,
            token=token,
            return_full_text=False,
            device_map="auto",
        )

        return pipe

    elif use == "mistral":

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
        tokenizer = AutoTokenizer.from_pretrained(model_name) # , padding_side="left")

        # padding_side='left'
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                # quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # trust_remote_code=True,
            )

        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer = tokenizer, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            return_full_text=False,
        )

        return pipe

    else:
        model_name = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token, 
            torch_dtype=torch.float16, 
            load_in_4bit=True,  
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            tokenizer=tokenizer,
            token=token,
            return_full_text=False,
            device_map="auto",
        )

        return pipe
    
# -----------------------------------------------------------------------
    
def perform_custom_task(LLM, data_dict, personas, task_prompt, task_name, possible_values, max_chars=2000, overwrite=False, verbose=1):

    flush()
    release_memory()

    for persona in tqdm_notebook(personas):
        if verbose:
            print("-"*20)
            print(persona)

        for qidx in tqdm_notebook(data_dict[persona].keys()):
            
            for depth in data_dict[persona][qidx].keys():

                for idx in data_dict[persona][qidx][depth].keys():

                    channel_name = data_dict[persona][qidx][depth][idx]["channel_name"]
                    video_title = data_dict[persona][qidx][depth][idx]["video_title"]
                    
                    if verbose:
                        print("Video title:", video_title, "Channel Name:", channel_name)

                    transcript = data_dict[persona][qidx][depth][idx]["transcript"]
                    transcript = ' '.join(transcript)
                    transcript = f"{channel_name}, {video_title}," + transcript

                    if len(transcript) < max_chars:
                        max_chars_ = len(transcript)
                    else:
                        max_chars_ = max_chars
                    
                    transcript = transcript[:max_chars_]

                    if verbose:
                        print(">> length of input", len(transcript))

                    # --------------------------

                    prompt = f"<s> [INST] {task_prompt} [/INST] Model answer </s> [INST] {transcript} [/INST]"

                    flush()
                    release_memory()

                    if (not task_name in data_dict[persona][qidx][depth][idx]) or overwrite: 

                        with torch.no_grad():

                            outputs = LLM(
                                prompt, 
                                max_new_tokens=128, 
                                do_sample=True, 
                                num_return_sequences=1,
                                temperature=0.25, top_k=50, top_p=0.95,
                            )

                        output = outputs[0]["generated_text"]
                        output = output.replace('"', "'")
                        
                        if verbose:
                            print(output)

                        data_dict[persona][qidx][depth][idx][task_name] = None
                        data_dict[persona][qidx][depth][idx][task_name + "_reasoning"] = None

                        res = None

                        for val in possible_values:

                            if f"[{val}]" in output:
                                res = val
                                break

                        reasoning = extract_pattern(output, pattern=r'\{.*\}')
    
                        if reasoning is None:
                            reasoning = extract_pattern(output, pattern=r'\{.*')

                        print("--------------------")
                        print("Answer:", res)
                        print("Reasoning:", reasoning)

                        if reasoning is None or res is None:
                            print(">>", output)

                        data_dict[persona][qidx][depth][idx][task_name] = res
                        data_dict[persona][qidx][depth][idx][task_name + "_reasoning"] = reasoning

# -----------------------------------------------------------------------
                        
def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

# -----------------------------------------------------------------------
    
def plot_histogram(personas, persona_queries, data_dict, persona_title_dict, max_depth, attribute_name="channel_name", threshold=0.25, figsize=(7, 12), y_label="", only_firsts=False):

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    attribute_per_persona = {}

    if only_firsts:
        max_depth = 1

    for pidx, persona in enumerate(personas):
        queries = persona_queries[persona]

        attribute_per_persona[persona] = []

        for qidx, _ in enumerate(queries): 
            for depth in range(0, max_depth):
                keys = data_dict[persona][qidx][depth].keys()

                for key in keys:
                    
                    attribute = data_dict[persona][qidx][depth][key][attribute_name]
                    attribute_per_persona[persona].append(attribute)

                    if only_firsts:
                        break

        attributes = attribute_per_persona[persona]

        ax = axes[pidx]

        string_counts = Counter(attributes)
        strings = list(string_counts.keys())
        counts = string_counts.values()

        sorted_counts_indices = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
        sorted_counts = [count for index, count in sorted_counts_indices]
        sorted_strings = [strings[index] for index, _ in sorted_counts_indices]

        sorted_counts = np.array(sorted_counts, dtype=np.float64)
        sorted_counts /= np.sum(sorted_counts)

        sum_percentage = 0

        for count_idx, count in enumerate(sorted_counts):
            sum_percentage += count
            if sum_percentage > threshold:
                break

        sorted_counts = list(sorted_counts)[:count_idx]
        sorted_strings = sorted_strings[:count_idx]

        sorted_strings = list(reversed(sorted_strings))
        sorted_counts = list(reversed(sorted_counts))

        sorted_counts = list(np.array(sorted_counts)*100)

        # Create histogram
        ax.barh(sorted_strings, sorted_counts)

        # Set labels and title
        ax.set_ylabel(y_label)
        ax.set_xlabel('Percentage (%)')

        ax.set_title(persona_title_dict[persona])

    title = "Top Channel Names by Frequency Coverage (30%)"

    if only_firsts:
        title += " (only firsts)"

    plt.suptitle(title, y=1.02)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------

def plot_grouped_bar_chart(ax, data_lists, labels, legend_fontsize=None, category_color_map=None):

    categories_counts = {}

    # Count categories for each list
    for i, data_list in enumerate(data_lists, start=1):
        categories_counts[f'List {i}'] = {}
        for category in data_list:
            if category not in categories_counts[f'List {i}']:
                categories_counts[f'List {i}'][category] = 1
            else:
                categories_counts[f'List {i}'][category] += 1

    # Extract categories and counts
    categories = sorted(set().union(*[categories_counts[key].keys() for key in categories_counts]))
    counts_lists = [[categories_counts[key].get(category, 0) for category in categories] for key in categories_counts]

    num_groups = len(labels)
    num_categories = len(categories)

    x = np.arange(num_groups)  # the label locations
    width = 0.2  # the width of the bars

    # Plot each group
    for i in range(num_categories):
        if category_color_map is not None:
            ax.bar(x + (i - 1) * width, [counts_list[i] for counts_list in counts_lists], width, label=categories[i], color=category_color_map[categories[i]], alpha=0.5)
        else:
            ax.bar(x + (i - 1) * width, [counts_list[i] for counts_list in counts_lists], width, label=categories[i], alpha=0.5)

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.5)

    if legend_fontsize is None:
        ax.legend(title='Metrics', loc="upper left")
    else:
        legend = ax.legend(fontsize=legend_fontsize, loc="upper left")
        legend.get_frame().set_alpha(0.5)  

# -----------------------------------------------------------------------

def plot_histogram_per_persona(personas, persona_queries, data_dict, category_color_map, task_names, max_depth, figsize=(6, 3)):

    _, axes = plt.subplots(1, 2, figsize=figsize)

    for tidx, task_name in enumerate(task_names):
        ax = axes[tidx]

        attributes_per_persona = []

        for persona in personas:
            queries = persona_queries[persona]

            attributes = []

            for qidx, _ in enumerate(queries): 
                for depth in range(0, max_depth):
                    keys = data_dict[persona][qidx][depth].keys()

                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]
                        if attribute != None:
                            attributes.append(attribute)

            attributes_per_persona.append(attributes)

        plot_grouped_bar_chart(ax, attributes_per_persona, personas, legend_fontsize=9, category_color_map=category_color_map)
        ax.set_title(task_name)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------
   
def plot_bar_chart(ax, values, x_tick_labels=None, xlabel='', ylabel='', title=''):
 
    x = range(len(values)) 
    
    ax.bar(x, values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if x_tick_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels) 
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(values) + 1)]) 
    
    ax.set_ylim(0, 1)  

# -------------------------------------------------------------------------

def plot_polarization_per_persona(personas, persona_queries, task_names, data_dict, max_depth, figsize=(6, 6), include_neutral=False):

    _, axes = plt.subplots(1, 2, figsize=figsize, sharey="row")

    for tidx, task_name in enumerate(task_names):

        attributes_per_persona = []

        for persona in personas:
            queries = persona_queries[persona]

            attributes = []

            for qidx, _ in enumerate(queries): 
                for depth in range(0, max_depth):
                    keys = data_dict[persona][qidx][depth].keys()

                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]
                        if attribute != None:
                            attributes.append(attribute)

            attributes_per_persona.append(attributes)

        entropy_per_persona = []

        for attribs in attributes_per_persona:
            attribs_ = []
            
            for attr in attribs:
                if attr != "neutral" or include_neutral:
                    attribs_.append(attr)

            entropy_per_persona.append(1.0 - calculate_entropy_normalized(attribs_))

        ax = axes[tidx]

        xlabel = "Persona"

        x_tick_labels = ["A", "B", "C"]

        plot_bar_chart(ax, entropy_per_persona, title=task_name, ylabel="Polarization", xlabel=xlabel, x_tick_labels=x_tick_labels)
        ax = axes[tidx]

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------

def plot_histogram_per_persona_per_depth(personas, persona_queries, data_dict, category_color_map, task_names, max_depth, figsize=(7, 12)):

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")

    attributes_dict = {}

    for tidx, task_name in enumerate(task_names):

        attributes_dict[tidx] = {}  

        for pidx, persona in enumerate(personas):
            queries = persona_queries[persona]
            attributes_dict[tidx][pidx] = {}  

            for depth in range(0, max_depth):
                attributes_dict[tidx][pidx][depth] = []

                for qidx, _ in enumerate(queries): 
                    keys = data_dict[persona][qidx][depth].keys()

                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]

                        if attribute is not None:
                            attributes_dict[tidx][pidx][depth].append(attribute)

    for row, task_name in enumerate(task_names):
        
        for col, persona in enumerate(personas):

            ax = axes[row, col]
            attribs_per_depth = []
            labels = []

            for depth in range(0, max_depth):
                labels.append(depth+1)
                attribs_per_depth.append(attributes_dict[row][col][depth])
            
            depth_labels = [f"{i+1}" for i in range(max_depth)]
            plot_grouped_bar_chart(ax, attribs_per_depth, depth_labels, legend_fontsize=9, category_color_map=category_color_map)

            if row == 0: 
                ax.set_title(persona)

            if col == 0:
                ax.set_ylabel(task_name)

            if row == len(task_names)-1:
                ax.set_xlabel("Depth")

            ax.xaxis.grid(False)
                
    plt.tight_layout()
    plt.suptitle("Metrics over depth", y=1.05)
    plt.show()

# -------------------------------------------------------------------------

def separate_into_bins(merged_list, number_of_bins=5):

    bin_size = len(merged_list) // number_of_bins
    bins = []

    current_bin_size = 0
    current_bin = []

    for item in merged_list:
        current_bin.append(item)
        current_bin_size += 1

        if current_bin_size == bin_size:
            bins.append(current_bin)
            current_bin = []
            current_bin_size = 0

    if current_bin:
        bins.append(current_bin)

    return bins

# -------------------------------------------------------------------------

def merge_lists(list_of_lists):
    new_list = []

    max_length = max(len(sublist) for sublist in list_of_lists)

    for i in range(max_length):
        for sublist in list_of_lists:
            if i < len(sublist):
                new_list.append(sublist[i])

    return new_list

# -------------------------------------------------------------------------

def plot_histogram_per_persona_per_bin(personas, persona_queries, data_dict, category_color_map, task_names, max_depth, figsize=(7, 12)):

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")

    attributes_dict = {}

    for tidx, task_name in enumerate(task_names):

        attributes_dict[tidx] = {}  

        for pidx, persona in enumerate(personas):
            queries = persona_queries[persona]
            attributes_dict[tidx][pidx] = []

            for qidx, _ in enumerate(queries): 

                per_depth = {}

                for depth in range(0, max_depth):
                    per_depth[depth] = []
                    keys = data_dict[persona][qidx][depth].keys()
                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]

                        if attribute is not None:
                            per_depth[depth].append(attribute)

                list_of_lists = []
                for k, v in per_depth.items():
                    list_of_lists.append(v)
                
                merged = merge_lists(list_of_lists)
                attributes_dict[tidx][pidx] += merged

    for row, task_name in enumerate(task_names):
        
        for col, persona in enumerate(personas):

            ax = axes[row, col]

            merged = attributes_dict[tidx][pidx]

            attribs_per_bin = separate_into_bins(merged)
            labels = [str(i+1) for i in range(0, 5)]

            attribs_per_bin = attribs_per_bin[:-1]

            plot_grouped_bar_chart(ax, attribs_per_bin, labels, legend_fontsize=9, category_color_map=category_color_map)

            if row == 0: 
                ax.set_title(persona)

            if col == 0:
                ax.text(-0.5, 0.5, task_name, va='center', rotation='vertical', fontweight="bold", transform=ax.transAxes)
                ax.set_ylabel("Counts")


            if row == len(task_names)-1:
                ax.set_xlabel("Bins")

            ax.xaxis.grid(False)
                
    plt.tight_layout()
    plt.suptitle("Distribution of metrics over time", y=1.05)
    plt.show()

# -------------------------------------------------------------------------

def plot_histogram_per_persona_per_depth_entropy(personas, persona_queries, data_dict, task_names, max_depth, figsize=(7, 12), include_neutral=True):

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")

    attributes_dict = {}

    for tidx, task_name in enumerate(task_names):

        attributes_dict[tidx] = {}  

        for pidx, persona in enumerate(personas):
            queries = persona_queries[persona]
            attributes_dict[tidx][pidx] = {}  

            for depth in range(0, max_depth):
                attributes_dict[tidx][pidx][depth] = []

                for qidx, _ in enumerate(queries): 
                    keys = data_dict[persona][qidx][depth].keys()

                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]

                        if attribute is not None:
                            if attribute != "neutral":
                                attributes_dict[tidx][pidx][depth].append(attribute)
    
    for row, task_name in enumerate(task_names):
        
        for col, persona in enumerate(personas):

            ax = axes[row, col]
            score_per_depth = []
            labels = []

            for depth in range(0, max_depth):
                labels.append(depth+1)

                score = 1.0 - calculate_entropy_normalized(attributes_dict[row][col][depth])
                score_per_depth.append(score)

            plot_bar_chart(ax, score_per_depth, title="", xlabel="")

            if row == 0: 
                ax.set_title(persona)

            if col == 0:
                ax.set_ylabel("Polarization")

                ax.text(-0.9, 0.5, task_name, va='center', rotation='vertical', fontweight="bold", transform=ax.transAxes)

            if row == len(task_names)-1:
                ax.set_xlabel("Bins")

            ax.xaxis.grid(False)
                
    plt.tight_layout()
    
    plt.suptitle("Polarization over time", y=1.0)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------

def plot_histogram_per_persona_per_bin_entropy(personas, persona_queries, data_dict, task_names, max_depth, figsize=(7, 12), include_neutral=True):

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey="row")

    attributes_dict = {}

    for tidx, task_name in enumerate(task_names):

        attributes_dict[tidx] = {}  

        for pidx, persona in enumerate(personas):
            queries = persona_queries[persona]
            attributes_dict[tidx][pidx] = []

            for qidx, _ in enumerate(queries): 

                per_depth = {}

                for depth in range(0, max_depth):
                    per_depth[depth] = []
                    keys = data_dict[persona][qidx][depth].keys()
                    for key in keys:
                        attribute = data_dict[persona][qidx][depth][key][task_name]

                        if attribute is not None:
                            per_depth[depth].append(attribute)

                list_of_lists = []
                for k, v in per_depth.items():
                    list_of_lists.append(v)
                
                merged = merge_lists(list_of_lists)
                attributes_dict[tidx][pidx] += merged

    for row, task_name in enumerate(task_names):
        
        for col, persona in enumerate(personas):

            ax = axes[row, col]

            merged = attributes_dict[tidx][pidx]

            merged_ = []

            for m in merged:
                if m != "neutral": #  or include_neutral:
                    merged_.append(m)

            attribs_per_bin = separate_into_bins(merged)
            labels = [str(i+1) for i in range(0, 5)]

            attribs_per_bin = attribs_per_bin[:-1]

            # ---------------

            score_per_bin = []
            labels = []

            for bin in attribs_per_bin:
                labels.append(depth+1)

                score = 1.0 - calculate_entropy_normalized(bin)
                score_per_bin.append(score)

            plot_bar_chart(ax, score_per_bin, title="", xlabel="")

            if row == 0: 
                ax.set_title(persona)

            if col == 0:
                ax.text(-0.9, 0.5, task_name, va='center', rotation='vertical', fontweight="bold", transform=ax.transAxes)
                ax.set_ylabel("Polarization")

            if row == len(task_names)-1:
                ax.set_xlabel("Bins")

            ax.xaxis.grid(False)
                
    plt.tight_layout()
    
    title = "Polarization Score over depth"
    
    if not include_neutral:
        title += " (without 'neutral)"

    plt.suptitle("Polarization over time", y=1.05)
    plt.show()