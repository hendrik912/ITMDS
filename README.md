## IT-management and Data Science 

Author: Hendrik Eilts

This code was created for the module "IT-management and Data Science" for our 'paper' with the title "Examining polarization on YouTube through personalized "sock
puppet" audits"

There 3 three main files:

### generate_queries.ipynb

In this file the generation of YouTube search queries is performed

### data_collection.py

Using the previously generated queries, data_collection.py performs the collection of data using a webcrawler based on https://gitlab.informatik.uni-bremen.de/hheuer/ratml 
Adding modifications to accept cookies, automated scrolling so the content is loaded, as well as loading the transcript (repeated presses on buttons)

### data_analysis.ipynb

In this file, we first extract the data we collected and then we perform classifications tasks using a LLM,
followed by plotting the results.



The required libraries for creating a virtual environment are requirements.txt
