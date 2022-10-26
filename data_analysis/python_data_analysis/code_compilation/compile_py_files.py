
import tempfile
import subprocess
from tqdm import tqdm
import argparse
from datasets import load_dataset


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Code compilation"
    )
    parser.add_argument(
        "--dataset_name",
        default="bigcode/python_permissive",
        type=str,
        help="HF repo name/path of the dataset.",
    )
    parser.add_argument(
        "--n_samples",
        default=10_000,
        type=int,
        help="Number of samples in the subset to analyze",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed",
    )
    return parser.parse_args()


def compile_python_code(sample):
    string = sample["content"]
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w") as f:
        f.write(string)
    py_command = "python{v} -m py_compile " + tmp.name
    
    try:
        subprocess.check_call(py_command.format(v=3).split())
        python3_works = True
    except subprocess.CalledProcessError:
        python3_works = False

    try:
        subprocess.check_call(py_command.format(v=2).split())
        python2_works = True
    except subprocess.CalledProcessError:
        python2_works = False

    return python2_works or python3_works


if __name__ == '__main__':
    args = parseArgs()

    print(f"Loading {args.n_samples} samples from {args.dataset_name} dataset")
    data = load_dataset(args.dataset_name, streaming=True, split="train", use_auth_token=True)
    subset = list(data.shuffle(seed=args.seed).take(args.n_samples))

    valid_files = 0
    for i in tqdm(range(len(subset))):
        if compile_python_code(subset[i]):
            valid_files += 1

    print(f"Number of valid python files in {args.n_samples} random samples: {valid_files}")
    print(f"Percentage of non valid files: {(len(subset) - valid_files) * 100 / len(subset)}%")