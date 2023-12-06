
import os
import joblib

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import youtube_crawler.utils as yc_utils

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

path_store_queries = os.path.join("data", "queries")

if __name__ == "__main__":

    # each persona is a collection of attributes in natural language
    personas = {
        "example" : "(Attributes: likes to cook, is male, is a middle aged US citizen)"
    }

    n = 5

    persona_query_dict = {}

    path_results = os.path.join(path_store_queries, model_name)

    for name, attributes in personas.items():


        if not os.path.exists(path_results):
            os.makedirs(path_results)

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        attributes = "(Attributes: likes to cook, is male, is a middle aged US citizen)"

        input = f"Impersonate a person that has the following attributes: {attributes}. " \
                f"and now give {n} queries you would search in youtube aligning with your personality."

        res = generator(
            input,
            max_length=50,
            num_return_sequences=2,
        )

        persona_query_dict[name] = res['generated_text']

        print(len(res))

    joblib.dump(persona_query_dict, os.path.join(path_results, "persona_query_dict"))