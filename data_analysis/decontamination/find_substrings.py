"""
Takes a direcrory containing jsonl files as input.
Filter out all samples that contain certain substrings.
"""
import sys
import os
import json
import glob
from tqdm import tqdm
from multiprocessing import Pool

from human_eval.data import read_problems
from datasets import load_dataset


# ========= data to filter out of the dataset ============
MBPP_PATH = "/data/mbpp/mbpp.jsonl"
TEST_IDS = list(range(11, 511))

def mbpp_docstrings():
    data = []
    with open(MBPP_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    
    data = [sample for sample in data if sample["task_id"] in TEST_IDS]

    assert len(data) == 500
        
    # Checksum / version issues here
    # dataset = load_dataset("mbpp", split="test")

    return [sample["text"] for sample in data]


def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        elif prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        else:
            raise ValueError()
    elif '\'\'\'' in prompt:
        assert prompt.count('\'\'\'') == 2
        return prompt.split('\'\'\'')[1].strip()
    else:
        raise ValueError()


def human_eval_docstrings():
    problems = read_problems()
    docstrings = [extract_docstring(v['prompt']) for k, v in problems.items()]
    return docstrings

FILTER_OUT = {
    "mbpp": mbpp_docstrings(),
    "human_eval": human_eval_docstrings()
}
# ============================================================

def add_dict(dict1: dict, dict2: dict) -> None:
    """
    Add the values of dict2 to dict1. All values must be int, float or dictionaries that also verify this condition.
    Will modify dict1 and return None
    """
    for key, value in dict2.items():
        if isinstance(value, (int, float)):
            if key not in dict1:
                dict1[key] = 0
            dict1[key] += value
        elif isinstance(value, dict):
            if key not in dict1:
                dict1[key] = {}
            assert isinstance(dict1[key], dict)
            add_dict(dict1[key], value)
        else:
            raise ValueError(f"Invalid type for key/value {key}: {value}")

def filter_file(data):
    """
    Return True, None if the file should be included in the dataset.
    Otherwise return False and some metadata about the file excluded
    """
    content = data['content'].lower()
    # For each substring, try to find it in the file (case insensitive)
    for benchmark, substrings in FILTER_OUT.items():
        for substring in substrings:
            if substring.lower() in content:
                return False, f"{benchmark}_match"

    # data, filter_reason = filter_content(data, FILTER_ARGS)
    # if data:
    #     return True, None
    # else:
    #     return False, filter_reason.value

    # Return True, None if none of the substrings was found
    return True, None


def _update_meta_dict(meta_dict, filter_reason):
    if filter_reason not in meta_dict:
        meta_dict[filter_reason] = 0
    meta_dict[filter_reason] += 1


def filter_jsonl_file(args):
    """
    Filter a given file and write the output to the disk
    """

    file_name, write_to = args
    meta = f"{write_to}_meta"
    meta_dict = {}
    with open(file_name, "r") as f:
        with open(write_to, "w") as out:
            with open(meta, "w") as meta_file:
                for i, line in tqdm(enumerate(f)):
                    data = json.loads(line)
                    # Write line to output-file if filter has passed
                    to_include, filter_reason = filter_file(data)
                    if to_include:
                        out.write(line)
                    else:
                        _update_meta_dict(meta_dict, filter_reason)
                # Dump meta dict
                meta_file.write(json.dumps(meta_dict))
                meta_file.write("\n")


def main():
    num_processes = 64
    # The input directory containing the jsonl files
    input_dir = sys.argv[1]
    # Where to write worker files and output file
    output_dir = sys.argv[2]

    assert os.path.isdir(input_dir)

    tmp_files_dir = os.path.join(output_dir, "tmp")
    output_file = os.path.join(output_dir, "data.jsonl")
    os.makedirs(tmp_files_dir, exist_ok=True)

    # Process all the files in the input directory
    # Get the arguments for each worker
    files = glob.glob(f"{input_dir}/data_*.jsonl")
    filter_args = [(file, f"{tmp_files_dir}/{os.path.basename(file)}") for file in files]
    output_files = [arg[1] for arg in filter_args]

    # Process the files in parallel
    with Pool(num_processes) as p:
        for i, res in enumerate(p.imap(filter_jsonl_file, filter_args)):
            print(i, res)
        
    # Concatenate the outputs of all the workers into one big file
    with open(output_file, "w") as outfile:
        for fname in output_files:
            with open(fname) as f:
                for line in f:
                    outfile.write(line)

    # compile meta
    meta = {}
    for fname in output_files:
        fmeta = json.load(open(f"{fname}_meta"))
        add_dict(meta, fmeta)
    with open(f"{output_file}_meta", "w") as outfile:
        json.dump(meta, outfile)


if __name__ == "__main__":
    main()

