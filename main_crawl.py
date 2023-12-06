
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import youtube_crawler.utils as yc_utils

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



if __name__ == "__main__":

    """
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    res = generator(
        "Hello World",
        max_length=30,
        num_return_sequences=2,
    )

    print(res)
    """

    NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD = 20
    NUMBER_OF_SEARCH_RESULTS_TO_CONSIDER = 10
    NUMBER_OF_RELATED_VIDEOS_TO_CONSIDER = 10
    NUMBER_OF_RELATED_VIDEOS_TO_VISIT_DEPTH = 4  # number of recommendations that are collected

    keywords = ["hello world"]

    yc_utils.crawl_youtube(keywords,
                           NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD, NUMBER_OF_SEARCH_RESULTS_TO_CONSIDER,
                           NUMBER_OF_RELATED_VIDEOS_TO_CONSIDER, NUMBER_OF_RELATED_VIDEOS_TO_VISIT_DEPTH,
                           "data", "hello_world")
