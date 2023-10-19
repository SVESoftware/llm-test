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
    print(f"History: last 4 messages (both user and human)")
    N = 4

    # Interactive questions and answers
    query = """Hello, I am writing here to inform you that I will write down more than 512 tokens so that we can
    have a test with it. So if Ivan and Bob are two people, where Ivan has 10 years and Bob 5 years. Which one is older?
    When will Ivan retire if he will retire 5 years after Bob and Bob will retire in 2035?
    """

    start = timeit.default_timer()
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """What's the time in Berlin now? And weather?"""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """Have you read a book called "Daemon" by Daniel Suarez?"""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """When was this book released?"""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """How many pages there?"""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """What is the best book you ever read?"""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")
    memory.chat_memory.messages = memory.chat_memory.messages[-N:]

    start = timeit.default_timer()
    query = """Program me function in python to flatten the list."""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")

    start = timeit.default_timer()
    query = """Do the same in Java."""
    print(f'\nQuestion: {query}')
    response = qa({'input': query})
    print(f'\nAnswer: {response["response"]}')
    end = timeit.default_timer()
    print(f"Time to answer: {end - start:.2f} s")

    print(f"Enter 'exit' to exit and 'clear' to clear the history from the context. Try 'show history'.")

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


