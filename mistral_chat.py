import box
import timeit
import yaml
import argparse
import logging
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_conversational_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup DBQA
    qa, memory = setup_conversational_dbqa()

    print(f"Running on: CPU")
    print(f"Display Source Documents set to: FALSE")
    print(f"Use history set to: YES")
    print(f"History: last 4 messages (2 from the Human and 2 from AI)")
    print(f"Type 'exit' to exit. 'clear' to clear session history.")
    N = 4


    while True:

        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query == "clear":
            memory.clear()


        start = timeit.default_timer()
        response = qa({'input': query})
        print(f'\nAnswer: {response["response"]}')
        end = timeit.default_timer()
        print(f"Time to answer: {end - start:.2f} s")


