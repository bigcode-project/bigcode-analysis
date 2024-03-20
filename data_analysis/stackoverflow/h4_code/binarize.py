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
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from h4.data.utils import save_dataset_shards


H4_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = H4_DIR / "data"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Added print statements / limit data size for debugging")
    parser.add_argument(
        "--output_dir",
        default=f"{DATA_DIR}/pmp-binarized",
        type=str,
        help="Where to save the processed dataset",
    )
    parser.add_argument(
        "--exchange_name",
        type=str,
        default=None,
        help="Optional argument to specify a specific subsection of the dataset",
    )
    parser.add_argument(
        "--binary_score", type=int, default=8, help="Score assigned to binarized pairs for preference data."
    )
    parser.add_argument(
        "--stream_data", action="store_true", help="Optionally stream data, which can be useful with weaker computers"
    )
    parser.set_defaults(debug=False, stream_data=False)  # default will process full dataset

    args = parser.parse_args()
    specific_exchange = args.exchange_name
    stream_dataset = args.stream_data
    binary_score = args.binary_score

    if specific_exchange:
        data_dir = "data/" + args.exchange_name
    else:
        data_dir = None

    if args.debug:
        data_len_limit = 10000
    else:
        data_len_limit = np.inf

    dataset = load_dataset(
        "HuggingFaceH4/pmp-stack-exchange",
        data_dir=data_dir,
        split="train",
        streaming=stream_dataset,
    )

    pmp_data = []
    for i, d in enumerate(iter(dataset)):
        # check debug limit, quit if in debug mode (don't save)
        if i > data_len_limit:
            print("Early exit for debug mode!")
            print(pmp_data)
            break

        question = d["question"]
        answers = d["answers"]
        num_answers = len(answers)

        answer_scores = [a["pm_score"] for a in answers]
        if len(np.unique(answer_scores)) < 2:
            print(f"PM Scores are {answer_scores}, skipping this question {i}")
        else:
            # Sample 2 unique scores for binarization
            dif_scores = False
            while not dif_scores:
                # print("infinite loop...?")
                two_answers = random.sample(answers, 2)

                if two_answers[0]["pm_score"] != two_answers[1]["pm_score"]:
                    dif_scores = True

        answer_0 = two_answers[0]
        answer_1 = two_answers[1]
        text_0 = "Question: " + question + "\n" + "Answer: " + answer_0["text"]
        text_1 = "Question: " + question + "\n" + "Answer: " + answer_1["text"]
        score_0 = binary_score
        score_1 = binary_score

        pmp_data.append({"context": text_0, "score": score_0})
        pmp_data.append({"context": text_1, "score": score_1})

    # Save binarized data
    sublist_len = 100000

    print(f"Dataset length is {len(pmp_data)}")
    # bypass known issue in arrow https://issues.apache.org/jira/browse/ARROW-17137
    print(f"Processed dataset length > {sublist_len}, processing to HF dataset in chunks")
    chunks = [pmp_data[x : x + sublist_len] for x in range(0, len(pmp_data), sublist_len)]
    ds_chunks = [Dataset.from_list(ch) for ch in chunks]
    ds = concatenate_datasets(ds_chunks)

    save_dataset_shards(ds, args.output_dir, subset="stackexchange", shard_size="100MB")
