from utils import crawler



if __name__ == "__main__":

    # Define queries that were generated with generate_queries.py

    # critical of immigration
    queries_persona_A = [
    "Why immigration is bad for America",
    "How immigrants are taking over our jobs",
    "Why immigration is a threat to our culture",
    "The truth about immigrant criminals",
    "The real reasons why immigrants come to the US",
    ]

    # supportive
    queries_persona_B = [
        "The impact of immigration on the US economy",
        "Immigrant entrepreneurs: success stories and lessons learned",
        "The role of immigration in shaping American culture",
        "Immigration reform: a comprehensive overview",
        "Immigration and the criminal justice system: a critical analysis",
    ]

    # neutral
    queries_persona_C = [
        "Pros and cons of immigration",
        "The impact of immigration on the economy",
        "Immigrant experiences in the UK",
        "Immigration and the NHS",
        "Immigration and education",
    ]

    # -----------------------------------
    # Hyperparameter for youtube crawler

    BROWSER = "firefox"

    NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD = 10
    NUMBER_OF_RELATED_VIDEOS_TO_CONSIDER = 10
    NUMBER_OF_RELATED_VIDEOS_TO_VISIT_DEPTH = 5

    data_folder = "data"
    names = ['Persona A', 'Persona B', 'Persona C']

    # Run youtube crawler for each set of queries
    for i, persona in enumerate([queries_persona_A, queries_persona_B, queries_persona_C]): 

        crawler.crawl_youtube(persona,
                            NUMBER_OF_PATHS_TO_COLLECT_PER_KEYWORD, 
                            NUMBER_OF_RELATED_VIDEOS_TO_CONSIDER,
                            NUMBER_OF_RELATED_VIDEOS_TO_VISIT_DEPTH,
                            data_folder, names[i], browser=BROWSER)
