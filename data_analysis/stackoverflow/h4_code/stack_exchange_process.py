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

from datasets import Dataset, concatenate_datasets

import py7zr
import requests
from h4.data.utils import save_dataset_shards


try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


H4_DIR = Path(__file__).resolve().parents[3]
# TODO: Ideally we would use PosixPath here, but it doesn't work with the way the script is implemented :)
DATA_DIR = str(H4_DIR) + "/data/pmp-stack-exchange/"

# stack exchanges we filter
ALL_EXCHANGES = [
    "3dprinting.meta",
    "3dprinting",
    "academia.meta",
    "academia",
    "ai.meta",
    "ai",
    "android.meta",
    "android",
    "anime.meta",
    "anime",
    "apple.meta",
    "apple",
    "arduino.meta",
    "arduino",
    "askubuntu",
    "astronomy",
    "astronomy.meta",
    "aviation",
    "aviation.meta",
    "avp",
    "avp.meta",
    "beer",
    "beer.meta",
    "bicycles",
    "bicycles.meta",
    "bioinformatics",
    "bioinformatics.meta",
    "biology",
    "biology.meta",
    "bitcoin",
    "bitcoin.meta",
    "blender",
    "blender.meta",
    "boardgames",
    "boardgames.meta",
    "bricks",
    "bricks.meta",
    "buddhism",
    "buddhism.meta",
    "cardano",
    "cardano.meta",
    "chemistry",
    "chemistry.meta",
    "chess",
    "chess.meta",
    "chinese",
    "chinese.meta",
    "christianity",
    "christianity.meta",
    "civicrm",
    "civicrm.meta",
    "codegolf",
    "codegolf.meta",
    "codereview",
    "codereview.meta",
    "coffee",
    "coffee.meta",
    "cogsci",
    "cogsci.meta",
    "computergraphics",
    "computergraphics.meta",
    "conlang",
    "conlang.meta",
    "cooking",
    "cooking.meta",
    "craftcms",
    "craftcms.meta",
    "crafts",
    "crafts.meta",
    "crypto",
    "crypto.meta",
    "cs",
    "cs.meta",
    "cseducators",
    "cseducators.meta",
    "cstheory",
    "cstheory.meta",
    "datascience",
    "datascience.meta",
    "dba",
    "dba.meta",
    "devops",
    "devops.meta",
    "diy",
    "diy.meta",
    "drones",
    "drones.meta",
    "drupal",
    "drupal.meta",
    "dsp",
    "dsp.meta",
    "earthscience",
    "earthscience.meta",
    "ebooks",
    "ebooks.meta",
    "economics",
    "economics.meta",
    "electronics",
    "electronics.meta",
    "elementaryos",
    "elementaryos.meta",
    "ell",
    "ell.meta",
    "emacs",
    "emacs.meta",
    "engineering",
    "engineering.meta",
    "english",
    "english.meta",
    "eosio",
    "eosio.meta",
    "esperanto",
    "esperanto.meta",
    "ethereum",
    "ethereum.meta",
    "expatriates",
    "expatriates.meta",
    "expressionengine",
    "expressionengine.meta",
    "fitness",
    "fitness.meta",
    "freelancing",
    "freelancing.meta",
    "french",
    "french.meta",
    "gamedev",
    "gamedev.meta",
    "gaming",
    "gaming.meta",
    "gardening",
    "gardening.meta",
    "genealogy",
    "genealogy.meta",
    "german",
    "german.meta",
    "gis",
    "gis.meta",
    "graphicdesign",
    "graphicdesign.meta",
    "ham",
    "ham.meta",
    "hardwarerecs",
    "hardwarerecs.meta",
    "health",
    "health.meta",
    "hermeneutics",
    "hermeneutics.meta",
    "hinduism",
    "hinduism.meta",
    "history",
    "history.meta",
    "homebrew",
    "homebrew.meta",
    "hsm",
    "hsm.meta",
    "interpersonal",
    "interpersonal.meta",
    "iot",
    "iot.meta",
    "iota",
    "iota.meta",
    "islam",
    "islam.meta",
    "italian",
    "italian.meta",
    "japanese",
    "japanese.meta",
    "joomla",
    "joomla.meta",
    "judaism",
    "judaism.meta",
    "korean",
    "korean.meta",
    "languagelearning",
    "languagelearning.meta",
    "latin",
    "latin.meta",
    "law",
    "law.meta",
    "lifehacks",
    "lifehacks.meta",
    "linguistics",
    "linguistics.meta",
    "literature",
    "literature.meta",
    "magento",
    "magento.meta",
    "martialarts",
    "martialarts.meta",
    "materials",
    "materials.meta",
    "math",
    "math.meta",
    "matheducators",
    "matheducators.meta",
    "mathematica",
    "mathematica.meta",
    "mathoverflow",
    "mechanics.meta",
    "mechanics",
    "meta.askubuntu",
    "meta.mathoverflow",
    "meta.serverfault",
    "meta.stackexchange",
    "meta.stackoverflow",
    "meta.superuser",
    "moderators.meta",
    "moderators",
    "monero.meta",
    "monero",
    "money.meta",
    "money",
    "movies.meta",
    "movies",
    "music.meta",
    "music",
    "musicfans.meta",
    "musicfans",
    "mythology.meta",
    "mythology",
    "networkengineering.meta",
    "networkengineering",
    "opendata.meta",
    "opendata",
    "opensource.meta",
    "opensource",
    "or.meta",
    "or",
    "outdoors.meta",
    "outdoors",
    "parenting.meta",
    "parenting",
    "patents.meta",
    "patents",
    "pets.meta",
    "pets",
    "philosophy.meta",
    "philosophy",
    "photo.meta",
    "photo",
    "physics.meta",
    "physics",
    "pm.meta",
    "pm",
    "poker.meta",
    "poker",
    "politics.meta",
    "politics",
    "portuguese.meta",
    "portuguese",
    "puzzling.meta",
    "puzzling",
    "quant.meta",
    "quant",
    "quantumcomputing.meta",
    "quantumcomputing",
    "raspberrypi.meta",
    "raspberrypi",
    "retrocomputing.meta",
    "retrocomputing",
    "reverseengineering.meta",
    "reverseengineering",
    "robotics.meta",
    "robotics",
    "rpg.meta",
    "rpg",
    "rus.meta",
    "rus",
    "russian.meta",
    "russian",
    "salesforce.meta",
    "salesforce",
    "scicomp.meta",
    "scicomp",
    "scifi.meta",
    "scifi",
    "security.meta",
    "security",
    "serverfault",
    "sharepoint",
    "sharepoint.meta",
    "sitecore",
    "sitecore.meta",
    "skeptics",
    "skeptics.meta",
    "softwareengineering",
    "softwareengineering.meta",
    "softwarerecs",
    "softwarerecs.meta",
    "sound",
    "sound.meta",
    "space",
    "space.meta",
    "spanish",
    "spanish.meta",
    "sports",
    "sports.meta",
    "sqa",
    "sqa.meta",
    "stackapps",
    "stats.meta",
    "stats",
    "stellar.meta",
    "stellar",
    "superuser",
    "sustainability",
    "sustainability.meta",
    "tex",
    "tex.meta",
    "tezos",
    "tezos.meta",
    "tor",
    "tor.meta",
    "travel",
    "travel.meta",
    "tridion",
    "tridion.meta",
    "ukrainian",
    "ukrainian.meta",
    "unix",
    "unix.meta",
    "ux",
    "ux.meta",
    "vegetarianism",
    "vegetarianism.meta",
    "vi",
    "vi.meta",
    "webapps",
    "webapps.meta",
    "webmasters",
    "webmasters.meta",
    "windowsphone",
    "windowsphone.meta",
    "woodworking",
    "woodworking.meta",
    "wordpress",
    "wordpress.meta",
    "workplace",
    "workplace.meta",
    "worldbuilding",
    "worldbuilding.meta",
    "writers",
    "writers.meta",
    "Stackoverflow",  # hardcoded for different URL structure
]

# Some excluded stack exchanges below (not a maintained list)
# spanish: es.meta.stackoverflow.com.7z, es.stackoverflow.com.7z
# japanese: ja.meta.stackoverflow.com.7z, ja.stackoverflow.com.7z
# some language: pt.stackoverflow.com, pt.meta.stackoverflow.com
# ru.stackoverflow, ru.meta.stackoverflow

# stack exchanges with different processing, these end in .net ;(
DOTNET_LIST = ["mathoverflow", "meta.mathoverflow"]

# stack exchanges without .stackoverflow.com (includes above)
SHORT_URL_LIST = [
    "askubuntu",
    "meta.askubuntu",
    "meta.serverfault",
    "meta.stackexchange",
    "meta.stackoverflow",
    "stackexchange",
    "superuser",
    "meta.superuser",
    "serverfault",
    "stackapps",
    "Stackoverflow",
]
SHORT_URL_LIST += DOTNET_LIST


def get_and_unpack_7z(directory: str, data_save_dir: str, save_dir_override: str = None):
    # check if unpacked data exists (no need to re-download):
    se_name_7z = directory[directory.rfind("/") + 1 :]
    se_name = se_name_7z[:-3]
    assert ".7z" == se_name_7z[-3:]
    if not os.path.exists(data_save_dir + se_name_7z):
        print("Loading raw data, this can take a second!")

        ex_data_url = (
            # "https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml/resolve/main/"\
            "https://archive.org/download/stackexchange/"
            + se_name_7z
        )

        response = requests.get(ex_data_url, allow_redirects=True)
        filename = os.path.basename(ex_data_url)

        print("Unpacking raw data.")
        if response.status_code == 200:
            with open(DATA_DIR + filename, "wb") as out:
                out.write(response.content)
            os.mkdir(DATA_DIR + se_name)
            with py7zr.SevenZipFile(DATA_DIR + filename, "r") as archive:
                if save_dir_override:
                    save_dir = save_dir_override
                else:
                    save_dir = se_name
                archive.extractall(DATA_DIR + save_dir + "/")
        else:
            print("Request failed: %d" % response.status_code)

        print("Loaded & unpacked data, now processing...")
    else:
        print("Raw 7z data already exists for this dir :)")


def print_dict(d):
    for key, val in d.items():
        print(f"{key}, {val}")


def simplify_date(date_string):
    date = datetime.datetime.strptime(date_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    return date.strftime("%Y/%m/%d")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="If the script will process all stack exchanges: warning, requires large amount of RAM",
    )
    parser.add_argument("--save_path", default=DATA_DIR, type=str, help="Path to the huggingface dataset preferably.")
    parser.add_argument(
        "--start_idx",
        default=0,
        type=int,
        help="Optional value to skip a number of exchanges in the above list if processing crashed midway",
    )
    parser.add_argument("--shard_size", default=100, type=int, help="Maximum size of file for subsets of data in MB")
    parser.add_argument("--debug", action="store_true", help="Added print statements for debugging")
    parser.set_defaults(debug=False, all=False)

    args = parser.parse_args()

    shard_size = str(args.shard_size) + "MB"
    process_all = args.all
    save_path = args.save_path
    start_idx = args.start_idx
    DEBUG = args.debug
    if process_all:
        se_list = ALL_EXCHANGES
    else:
        print("Run from command line with --all=True to process all data")
        se_list = ["ai", "apple", "pets", "ai.meta"]

    os.makedirs(DATA_DIR, exist_ok=True)

    # Process all exchanges in loop (saves in memory)
    TOTAL = len(se_list) - 1
    for i, se_sub_name in enumerate(se_list[start_idx:]):
        print(f"SECTION {i + start_idx}/{TOTAL}: {se_sub_name} - START")

        # some stack exchanges don't use .stackexchange.com
        if se_sub_name not in SHORT_URL_LIST:
            se_full_name = se_sub_name + ".stackexchange.com"
        elif se_sub_name in DOTNET_LIST:  # two exchanges need .net
            se_full_name = se_sub_name + ".net"
        else:
            se_full_name = se_sub_name + ".com"

        start_time = time.time()
        full_section_data = []

        # https://archive.org/download/stackexchange/Stackoverflow.com-Posts.7z
        # https://archive.org/download/stackexchange/Stackoverflow.com-Users.7z

        # get_and_unpack_7z()
        ex_data_file = DATA_DIR + se_full_name + "/Users.xml"
        # check if unpacked data exists:
        if not os.path.exists(ex_data_file):
            # get raw data
            ex_data_file_7z = se_full_name + ".7z"
            if "Stackoverflow.com" in ex_data_file_7z:
                base_stackoverflow_dir = ex_data_file_7z[:-3]
                get_and_unpack_7z(
                    base_stackoverflow_dir + "-Posts.7z", DATA_DIR, save_dir_override="stackoverflow.com"
                )
                get_and_unpack_7z(
                    base_stackoverflow_dir.lower() + "-Users.7z", DATA_DIR, save_dir_override="stackoverflow.com"
                )  # users dir only is lowercase s
            else:
                get_and_unpack_7z(ex_data_file_7z, DATA_DIR)

        # load extracted xml files
        local_path = (
            DATA_DIR + se_full_name.lower() + "/"
        )  # "ai.stackexchange.com/" # again, .lower() for the Stackexchange.com/Users
        posts_subpath = "Posts.xml"
        users_subpath = "Users.xml"

        """
        XML file structure:
        * PostTypeID ranges from 1: Question, 2: Answer, ....
        * We only want posts with AcceptedAnswerId fields
        (docs https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede)
        """

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
                            "https://" + se_full_name + "/questions/" + str(tag),  # question URL
                            "https://" + se_full_name,  # Exchange URL
                            "https://"
                            + se_full_name
                            + "/users/"
                            + str(user_id)
                            + "/",  # Author URL -- don't include username afterwards to avoid case with spaces in name (string regex problem)
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
            for exchange in tree.iter("row"):
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
                        # fix rare case that the username for answer authors is not in the database
                        if user_id in user_info:
                            username = user_info[user_id]
                        else:
                            username = "(user-not-found)"
                        answer_info[parent]["AuthorNames"].append(username)

                        if DEBUG:
                            print_dict(answer_info[parent])

        qa_keys = question_info.keys()

        final_outputs = {"domain": se_sub_name}

        for k in qa_keys:
            question_data = question_info[k]

            accepted_ans = question_data["AcceptedAnswerId"]

            answer_data = answer_info[k]
            metadata = question_data["metadata"]
            date = question_data["Date"]

            # filter for number of unique scores to be >= 2 (per paper)
            scores = answer_data["Score"]
            if len(np.unique(scores)) >= 2:
                answers = []
                for text, score, ans_id, auth_name, auth_id in zip(
                    answer_data["Text"], scores, answer_data["Id"], answer_data["AuthorNames"], answer_data["Author"]
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

                    sub_answer["answer_id"] = ans_id
                    sub_answer["text"] = text
                    sub_answer["pm_score"] = s
                    sub_answer["selected"] = accepted
                    sub_answer["author"] = auth_name
                    sub_answer["author_id"] = auth_id
                    sub_answer["author_profile"] = "https://" + se_full_name + "/users/" + str(auth_id)
                    answers.append(sub_answer)

            json_obj = {
                "qid": k,
                "question": question_data["Body"],
                "answers": answers,
                "date": date,
                "metadata": metadata,
            }
            full_section_data.append(json_obj)

        print(f"finished section {se_full_name} at {time.time() - start_time}s")

        if not DEBUG:
            sublist_len = 100000

            # bypass known issue in arrow https://issues.apache.org/jira/browse/ARROW-17137
            if len(full_section_data) > sublist_len:
                print(f"Processed dataset length > {sublist_len}, processing to HF dataset in chunks")
                chunks = [
                    full_section_data[x : x + sublist_len] for x in range(0, len(full_section_data), sublist_len)
                ]
                ds_chunks = [Dataset.from_list(ch) for ch in chunks]
                ds = concatenate_datasets(ds_chunks)
            else:
                ds = Dataset.from_list(full_section_data)

            save_dataset_shards(ds, save_path, subset=se_full_name, shard_size=shard_size)
