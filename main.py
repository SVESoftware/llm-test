import box
import timeit
import yaml
import argparse
import logging
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_c_dbqa, setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    start = timeit.default_timer()

    # Setup DBQA
    qa = setup_c_dbqa()

    print(f"Running on: CPU")
    print(f"Display Source Documents set to: FALSE")
    print(f"Use history set to: YES")

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        #if not query.endswith("?"):
        #    query = query + "?"
        #print(f"Query: {query}")
        # Get the answer from the chain

        response = qa({'input': query})
        print(f'\nAnswer: {response}')
        end = timeit.default_timer()
        print(f"Time to answer: {end - start:.2f} s")
