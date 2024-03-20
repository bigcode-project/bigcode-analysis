# Inspired by https://github.com/huggingface/h4/blob/main/scripts/data/pmp/stack_exchange_process.py
import datetime
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# Note: Using rclone + py7zr in command line is often faster than this
import py7zr
import requests

# If the cleaning becomes a bottleneck at some point, could be better to use
# this snippet from Anton https://gist.github.com/anton-l/4bfafb42878a8e77b20f3b844d9cae36
# (uses selectolax, faster than bs4) instead.
from bs4 import BeautifulSoup
from se_reference_utils import ALL_EXCHANGES


DATA_DIR = "data/stack-exchange"
WTOKEN = os.getenv("WTOKEN")


def simplify_date(date_string):
    date = datetime.datetime.strptime(date_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    return date.strftime("%Y/%m/%d")


def download_and_extract_se7z(name: str, directory: str, data_save_dir: str, save_dir_override: str = None):
    # Downloading 7z file
    if os.path.exists(f"{data_save_dir}/{name}.7z"):
        print("Raw 7z data already exists for this dir.")
    else:
        print("Downloading compressed data.")

        ex_data_url = f"https://archive.org/download/stackexchange/{directory}"
        response = requests.get(ex_data_url, allow_redirects=True)

        if response.status_code != 200:
            raise ConnectionError(f"Request failed: {response.status_code} for subset: {name}, url: {ex_data_url}")

        print("Unpacking raw data.")
        with open(f"{DATA_DIR}/{name}.7z", "wb") as out:
            out.write(response.content)

    os.mkdir(f"{DATA_DIR}/{name}")
    with py7zr.SevenZipFile(f"{DATA_DIR}/{name}.7z", "r") as archive:
        save_dir = save_dir_override if save_dir_override is not None else name
        archive.extractall(f"{DATA_DIR}/{save_dir}/")

    print(f"{name} successfully extracted.")


def get_question_from_html(exchange):
    question = {}
    keys_of_interest = ["Id", "Body", "AnswerCount", "OwnerUserId", "PostScore", "Date", "AcceptedAnswerId"]
    for key in keys_of_interest:
        try:
            if key in ["Id", "AnswerCount", "PostScore", "AcceptedAnswerId", "OwnerUserId"]:
                question[key] = int(exchange.attrib[key])
            elif key == "Date":
                question[key] = simplify_date(exchange.attrib["CreationDate"])
            elif key == "Body":
                question[key] = exchange.attrib[key]
                question["text"] = BeautifulSoup(exchange.attrib[key], "lxml").text
            else:
                question[key] = exchange.attrib[key]
        except KeyError:
            # deleted user redirect to community page > -1
            question[key] = -1 if key == "OwnerUserId" else None

    question["metadata"] = [
        f"https://{se_sub_url}/questions/{str(question['Id'])}",  # question URL
        f"https://{se_sub_url}",  # Exchange URL
        f"https://{se_sub_url}/users/{str(question['OwnerUserId'])}/",  # Author URL
    ]

    return question["Id"], question


def get_answer_from_html(exchange):
    # We connect answers to their parent's id
    parent_id = int(exchange.attrib["ParentId"])

    answer = {}
    keys_of_interest = ["Body", "Score", "Id", "OwnerUserId"]
    for key in keys_of_interest:
        try:
            if key in ["Score", "Id", "OwnerUserId"]:
                answer[key] = int(exchange.attrib[key])
            elif key == "Body":
                answer[key] = exchange.attrib[key]
                answer["text"] = BeautifulSoup(exchange.attrib[key], "lxml").text
            else:
                answer[key] = exchange.attrib[key]
        except KeyError:
            answer[key] = -1 if key == "OwnerUserId" else None

    return parent_id, answer


def get_posts_from_html(se_sub_name):
    extracted_info = defaultdict(lambda: {"question": None, "answers": list()})
    with open(f"{DATA_DIR}/{se_sub_name}/Posts.xml", "rb") as f:
        tree = ET.parse(f)

        for exchange in tree.iter("row"):
            post_type = int(exchange.attrib["PostTypeId"])

            if post_type == 1:  # Question
                if int(exchange.attrib["AnswerCount"]) > 0:
                    tag, question = get_question_from_html(exchange)
                    extracted_info[tag]["question"] = question

            elif post_type == 2:  # Answer
                tag, answer = get_answer_from_html(exchange)
                extracted_info[tag]["answers"].append(answer)
    return extracted_info


def get_jsonlines_from_posts(extracted_info):
    result_jsonlines = []
    for tag, data in extracted_info.items():
        # Sorting answers by score (see LLAMA paper), and only keep positively scored ones
        question = data["question"]
        answers = [a for a in sorted(data["answers"], key=lambda x: x["Score"]) if a["Score"] > 0]

        # We skip empty questions or answers
        if question is None or len(answers) < 1:
            continue

        text = f"user{question['OwnerUserId']}: {question['text']}"
        for answer in answers:
            text += f"\nuser{answer['OwnerUserId']}: {answer['text']}"

        result = {
            "question_id": question["Id"],
            "text": text,
            "metadata": question["metadata"],
            "date": question["Date"],
            "original_text": [f"{item['OwnerUserId']}: {item['Body']}" for item in [question] + answers],
        }
        result_jsonlines.append(result)
    return result_jsonlines


def upload_to_hub(result_jsonlines):
    size = len(result_jsonlines)
    chunk_size = 100000
    if size > chunk_size:
        chunks = [
            Dataset.from_list(result_jsonlines[i : min(i + chunk_size, size)]) for i in range(0, size, chunk_size)
        ]
        dataset = concatenate_datasets(chunks)
    else:
        dataset = Dataset.from_list(result_jsonlines)

    dataset.push_to_hub("HuggingFaceGECLM/StackExchange_Mar2023", split=se_sub_name, private=True, token=WTOKEN)


def main(se_sub_name, se_sub_url):
    print(f"{se_sub_name} at {se_sub_url}.")
    start_time = time.time()

    # Download and extract
    if not os.path.exists(f"{DATA_DIR}/{se_sub_name}/Posts.xml"):
        if "se_sub_name" == "stackoverflow":
            # Note: we'll also need -Users.7z if we want to filter on licenses at some point
            download_and_extract_se7z(
                se_sub_name, f"{se_sub_url}-Posts.7z", DATA_DIR, save_dir_override="stackoverflow.com"
            )
        else:
            download_and_extract_se7z(se_sub_name, f"{se_sub_url}.7z", DATA_DIR)

    # Selects posts from HTML tree (Questions and answers)
    extracted_info = get_posts_from_html(se_sub_name)
    print("Posts parsed from HTML.")

    # Create json from posts
    result_jsonlines = get_jsonlines_from_posts(extracted_info)

    print(f"Finished {se_sub_url} in {time.time() - start_time}s. Contains {len(result_jsonlines)} lines.")

    # Saves to the hub
    upload_to_hub(result_jsonlines)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    # Process all exchanges in a loop - could be easily launched in parallel
    for se_sub_name, se_sub_url in tqdm(ALL_EXCHANGES.items()):
        main(se_sub_name, se_sub_url)
