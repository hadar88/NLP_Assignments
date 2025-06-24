# BGU CS Course 'NLP with LLMs' - Spring 2025 - Michael Elhadad - HW2
## June 2025

This repository contains instructions for Assignment 2 in the course.

To setup your environment, the prerequisites are:
* python (> 3.11)
* git
* uv https://docs.astral.sh/uv/getting-started/)
* Visual studio code 

To setup the environment do:

1. Create a folder for the assignment: mkdir hw2; cd hw2
2. Retrieve the dataset we will use and the code from this repo:
    2. git clone https://github.com/melhadad/nlp-with-llms-2025-hw2.git
3. Load the required python libraries:
    1. cd nlp-with-llms-2025-hw2; uv sync
4. Define your API keys in grok_key.ini
    1. Define the environment variables in your shell - for example:
    ```
    # Unix like
    source grok_key.ini
    export XAI_API_KEY=$XAI_API_KEY
    ```
5. Activate the project virtual env: 
   ```
   source .venv\bin\activate
   ```
6. Open the notebooks (*.ipynb) in VS Code and verify you can execute the cells.
    ```
    code .
    ```


