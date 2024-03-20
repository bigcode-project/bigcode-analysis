# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import os
import time


try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from argparse import ArgumentParser

import numpy as np


parser = ArgumentParser()
parser.add_argument("--stack_exchange", default="ai", type=str, help="Which stack exchange data to process")
parser.add_argument(
    "--save_to_text", default=False, type=bool, help="Whether or not the outputs are saved to a text file."
)
parser.add_argument("--debug", default=False, type=bool, help="Added print statements for debugging")

args = parser.parse_args()

save = args.save_to_text
se_name = args.stack_exchange + ".stackexchange.com"
DEBUG = args.debug


start_time = time.time()

data_dir = "data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# check if unpacked data exists:
ex_data_file = data_dir + se_name + "/Posts.xml"
if not os.path.exists(ex_data_file):
    # get raw data
    ex_data_file_7z = se_name + ".7z"
    if not os.path.exists(data_dir + ex_data_file_7z):
        print("Loading raw data, this can take a second!")
        import py7zr
        import requests

        ex_data_url = (
            "https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml/resolve/main/"
            + ex_data_file_7z
        )
        response = requests.get(ex_data_url, allow_redirects=True)
        filename = os.path.basename(ex_data_url)

        if response.status_code == 200:
            with open(data_dir + filename, "wb") as out:
                out.write(response.content)
            os.mkdir(data_dir + se_name)
            with py7zr.SevenZipFile(data_dir + filename, "r") as archive:
                archive.extractall(data_dir + se_name + "/")
        else:
            print("Request failed: %d" % response.status_code)

        print("Loaded data, now processing!")

# load extracted xml files
local_path = data_dir + se_name + "/"  # "ai.stackexchange.com/"
posts_subpath = "Posts.xml"
votes_subpath = "Votes.xml"
users_subpath = "Users.xml"

"""
XML file structure:
* PostTypeID ranges from 1: Question, 2: Answer, ....
* We only want posts with AcceptedAnswerId fields

(docs https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede)
"""


def print_dict(d):
    for key, val in d.items():
        print(f"{key}, {val}")


def simplify_date(date_string):
    date = datetime.datetime.strptime(date_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    return date.strftime("%Y/%m/%d")


user_info = {-1: "(user-deleted)"}
question_info = {}
answer_info = {}

# extract user data for license
with open(local_path + users_subpath, "rb") as f:  # Users file
    tree = ET.parse(f)
    for exchange in tree.iter("row"):
        tag = int(exchange.attrib["Id"])
        user_info[tag] = str(exchange.attrib["DisplayName"])

if DEBUG:
    print_dict(user_info)

with open(local_path + posts_subpath, "rb") as f:  # Posts file
    tree = ET.parse(f)

    # process questions, find answers next
    # note, could do this all in one loop and store anything is memory is cheaper than processing speed

    # iterator through all rows
    for exchange in tree.iter("row"):
        # find 2+ answers
        if "AnswerCount" in exchange.attrib:
            ans_count = int(exchange.attrib["AnswerCount"])

            # only save questions with >= 2 answers
            if ans_count >= 2:
                tag = int(exchange.attrib["Id"])

                result = {}
                result["Body"] = exchange.attrib["Body"]

                # store some metadata
                result["AnswerCount"] = ans_count
                result["PostScore"] = int(exchange.attrib["Score"])

                # save metadata
                if "OwnerUserId" in exchange.attrib:
                    user_id = int(exchange.attrib["OwnerUserId"])
                else:
                    user_id = -1  # deleted user redirect to community page

                result["Author"] = user_id  # should fail for some deleted entries
                result["metadata"] = [
                    "https://" + se_name + "/questions/" + str(tag),
                    "https://" + se_name,
                    "https://"
                    + se_name
                    + "/users/"
                    + str(user_id)
                    + "/",  # don't include username afterwards to avoid case with spaces in name (string regex problem)
                ]
                result["Date"] = simplify_date(exchange.attrib["CreationDate"])

                # if accepted answer, store it
                if "AcceptedAnswerId" in exchange.attrib:
                    accepted_ans = int(exchange.attrib["AcceptedAnswerId"])
                    result["AcceptedAnswerId"] = accepted_ans
                else:
                    result["AcceptedAnswerId"] = None

                question_info[tag] = result
                if DEBUG:
                    print_dict(question_info[tag])

    # process looking for answers
    for i, exchange in enumerate(tree.iter("row")):
        # answers are ID type 2
        if int(exchange.attrib["PostTypeId"]) == 2:
            # get parent, check if in question_info
            parent = int(exchange.attrib["ParentId"])
            # note, that parent will be same as tag above in answer_info and question_info

            # log if parent is in questions (multiple answers for preference model)
            if parent in question_info:
                # info for answers
                ans_text = exchange.attrib["Body"]
                ans_score = int(exchange.attrib["Score"])
                ans_id = int(exchange.attrib["Id"])  # extra score if this ID matches accept id above

                # save metadata
                if "OwnerUserId" in exchange.attrib:
                    user_id = int(exchange.attrib["OwnerUserId"])
                else:
                    user_id = -1  # deleted user
                # we'll need to store multiple answers per tag
                if parent not in answer_info:
                    answer_info[parent] = {}
                    answer_info[parent]["Text"] = []
                    answer_info[parent]["Score"] = []
                    answer_info[parent]["Id"] = []
                    answer_info[parent]["Author"] = []
                    answer_info[parent]["AuthorNames"] = []

                answer_info[parent]["Text"].append(ans_text)
                answer_info[parent]["Score"].append(ans_score)
                answer_info[parent]["Id"].append(ans_id)
                answer_info[parent]["Author"].append(user_id)  # should fail for some deleted entries
                answer_info[parent]["AuthorNames"].append(user_info[user_id])

                if DEBUG:
                    print_dict(answer_info[parent])

# don't debug and save
if DEBUG:
    quit()

qa_keys = question_info.keys()
if save:
    import json

    output_file = open(data_dir + "output.jsonl", "w")

final_outputs = {"domain": args.stack_exchange}
print(" ------ printing processed questions ------ ------ ------ ------ ------ ------ ")
for k in qa_keys:
    question_data = question_info[k]
    if not save:
        print("  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ")
        print(f"Question (id: {k}): {question_data['Body']}")

    accepted_ans = question_data["AcceptedAnswerId"]

    answer_data = answer_info[k]
    metadata = question_data["metadata"]
    date = question_data["Date"]
    # filter for number of unique scores to be >= 2 (per paper)
    scores = answer_data["Score"]
    if len(np.unique(scores)) >= 2:
        answers = []
        for i, (text, score, ans_id, auth_name, auth_id) in enumerate(
            zip(answer_data["Text"], scores, answer_data["Id"], answer_data["AuthorNames"], answer_data["Author"])
        ):
            sub_answer = {}
            accepted = accepted_ans == ans_id

            if score >= 0:
                s = round(np.log2(1 + score))

                # not documented if negative answers can be accepted, assuming no
                if accepted:  # add 1 to score if answer was accepted
                    s += 1
            else:
                s = -1

            # print or save, *** indicates preferred answer
            pref = ", ***" if accepted else ""
            sub_answer["AnswerID"] = ans_id
            sub_answer["text"] = text
            sub_answer["pm_score"] = s
            sub_answer["selected"] = accepted
            sub_answer["Author"] = auth_name
            sub_answer["AuthorID"] = auth_id
            sub_answer["AuthorProfile"] = "https://" + se_name + "/users/" + str(auth_id)
            answers.append(sub_answer)
            if not save:
                print(f"Answer (id {ans_id}, s:{s}{pref}): {text}")
                print("  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ")

        if save:
            json_obj = {
                "qid": k,
                "question": question_data["Body"],
                "answers": answers,
                "date": date,
                "metadata": metadata,
            }
            json.dump(json_obj, output_file)

print(f"finished at {time.time() - start_time}s")
"""
Added options/notes for scaling & changing this script

Adding a dataloader to use HuggingFace Datasets
`from datasets import load_dataset`
-----

Logs on loading 7z files:
Example for samsum dataset::
https://github.com/huggingface/datasets/blob/fedf891a08bfc77041d575fad6c26091bc0fce52/datasets/samsum/samsum.py#L106-L110
-----

Making a cleaner repo + dataloader out of the raw data here:
https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml/tree/main
* move many files into folder (how to do that without loading)?
* add data loader (see above, shouldn't be so hard)
* figure out storage datatype of the processed data
----

Maybe consider using Beautiful Soup?
https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# list files in the raw repository
from huggingface_hub import HfApi
api = HfApi()

se_files = api.list_repo_files("flax-sentence-embeddings/stackexchange_xml", repo_type="dataset")
se_data_files = [f for f in se_files if "7z" in f]
se_names = [f[:f.find(".")] for f in se_files if "7z" in f]
se_names = [f + ".meta" if (i%2) == 0 else f for i, f in enumerate(se_names)]
# print(se_data_files)

"""
