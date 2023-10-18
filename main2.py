import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_c_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='How much is the minimum guarantee payable by adidas?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    # Setup DBQA
    qa = setup_c_dbqa()
    chat_history = []
    query = "How old is the first employee?"
    print("PITANJE")
    response = qa({'query': query, 'history': chat_history})
    print(f'\nAnswer: {response["result"]}')

    end = timeit.default_timer()
    print(f"Time to retrieve response: {end - start}")
    start = timeit.default_timer()

    chat_history.append((query, response["result"]))
    print("PITANJE")
    query = "What is his name?"
    response = qa({'query': query, "history": chat_history})
    print(f'\nAnswer: {response["result"]}')

    chat_history = chat_history[-1:]

    chat_history.append((query, response["result"]))
    print("PITANJE")
    query = "Which one is older? Pero or Ivica?"
    response = qa({'query': query, "history": chat_history})
    print(f'\nAnswer: {response["result"]}')

    end = timeit.default_timer()
    print(f"Time to retrieve response: {end - start}")
    start = timeit.default_timer()

    chat_history = chat_history[-1:]

    chat_history.append((query, response["result"]))
    print("PITANJE")
    query = "Ivica will retire in 10 years. Which year will it be when this happens?"
    response = qa({'query': query, "history": chat_history})
    print(f'\nAnswer: {response["result"]}')

    end = timeit.default_timer()
    print(f"Time to retrieve response: {end - start}")
    start = timeit.default_timer()


    chat_history = chat_history[-1:]

    chat_history.append((query, response["result"]))
    print("PITANJE")
    query = "Let's assume that current year is 2025 and Pero will retire in 30 years. When will Pero retire?"
    response = qa({'query': query, "history": chat_history})
    print(f'\nAnswer: {response["result"]}')

    end = timeit.default_timer()
    print(f"Time to retrieve response: {end - start}")
    start = timeit.default_timer()

    chat_history = chat_history[-1:]

    chat_history.append((query, response["result"]))
    print("PITANJE")
    query = "If we add 10 to that year, which year it will be? Is this a leap year?"
    response = qa({'query': query, "history": chat_history})
    print(f'\nAnswer: {response["result"]}')

    end = timeit.default_timer()
    print(f"Time to retrieve response: {end - start}")
    start = timeit.default_timer()







