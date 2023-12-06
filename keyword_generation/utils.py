
import os
import joblib
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import youtube_crawler.utils as yc_utils


# -------------------------------------------------------------------------------------------

def generate_keywords(model, tokenizer, personas, number_of_queries_to_generate=5):

    persona_query_dict = {}

    for name, attributes in personas.items():

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        attributes = "(Attributes: likes to cook, is male, is a middle aged US citizen)"

        input = f"Impersonate a person that has the following attributes: {attributes} " \
                f"and now create a queries you would search in youtube aligning with your personality."

        res = generator(
            input,
            max_length=50,
            num_return_sequences=number_of_queries_to_generate,
        )

        res_list = []
        for r in res:
            res_list.append(r['generated_text'])

        persona_query_dict[name] = res_list

    return persona_query_dict

# -------------------------------------------------------------------------------------------
