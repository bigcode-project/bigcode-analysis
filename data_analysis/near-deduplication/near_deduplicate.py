import json
import os
import time
from pathlib import Path
import re
from huggingface_hub import Repository
from multiprocessing import Pool
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

from datasets import load_dataset

from minhash_deduplication import deduplicate_dataset


def parse_args():
    parser = ArgumentParser(description='near deduplication')
    parser.add_argument(
            "--dataset_name",
            default="bigcode-data/python_any_license_v2",
            type=str,
            help="dataset to deduplicate, path to HF repo or local path",
        )
    parser.add_argument(
            "--text_column",
            default="content",
            type=str,
            help="column name of the text to dedulicate",
        )
    parser.add_argument(
            "--num_workers",
            default=96,
            type=int,
            help="number of workers for deduplication",
        )
    parser.add_argument(
            "--jaccard_threshold",
            default=0.85,
            type=float,
            help="Jaccard similarity threshold",
        )
    # we save data locally before pushing to the Hub to avoid any issues
    # the remote HF repo where we want the new data is cloned inside a folder out_path 
    # and the data is saved inside
    parser.add_argument(
            "--repo_name",
            default="python_any_license_v2_near_dedup",
            type=str,
            help="HF repo where deduplicated dataset will be pushed later, repo is cloned, and data is saved inside",
        )
    parser.add_argument(
            "--out_path",
            default="./data/data-near-dedup",
            type=str,
            help="local directory where repo_name is cloned",
        )
    parser.add_argument(
            "--org",
            default="bigcode-data",
            type=str,
            help="HF org/username where the data will be pushed",
        )
    parser.add_argument(
            "--shard_size",
            default=1000 << 20,
            type=int,
            help="size of the dataset shards",
        )
    parser.add_argument(
            "--test_run",
            default=False,
            type=bool,
            help="make a test run, if True we only deduplicate a small subset",
        )
    return parser.parse_args()



def save_shard(shard_tuple):
    """Save shard"""
    filename, shard = shard_tuple
    shard.to_parquet(filename)

args = parse_args()

print("setting up the repo")
repo = Repository(
        local_dir=args.out_path,
        clone_from=args.org + "/" + args.repo_name,
        repo_type="dataset",
        private=True,
        use_auth_token=True,
        git_user=args.org
        )
output_dir = Path(args.out_path)
output_dir.mkdir(exist_ok=True)
os.mkdir(args.out_path + "/data")
print("setup done")


t_start = time.time()
# the data is saved in the cache for future loadings
ds = load_dataset(args.dataset_name, split="train", use_auth_token=True) 
#ds = load_dataset("bigcode-data/python_any_license_v2", split="train", use_auth_token=True)

if args.test_run:
    # for a test run we only use a small subset
    ds = ds.select([i for i in range(7000)])
init_size = len(ds)
print(f"Time to load dataset: {time.time()-t_start:.2f}")


# Deduplicate with minhash and jaccard similarity
t_start = time.time()
ds, duplicate_clusters = deduplicate_dataset(ds, args.jaccard_threshold)
new_size = len(ds)
print(f"Time to deduplicate dataset: {time.time()-t_start:.2f}")
print(f"Size of deduplicated dataset: {len(ds)}, old dataset size {init_size}")
with open("size_info.json", "w") as f:
    json.dump([init_size, new_size, (init_size-new_size)*100/init_size],f)


with open(output_dir / "duplicate_clusters.json", "w") as f:
    json.dump(duplicate_clusters, f)


if ds._indices is not None:
    dataset_nbytes = ds.data.nbytes * len(ds._indices) / len(ds.data)
else:
    dataset_nbytes = ds.data.nbytes
num_shards = int(dataset_nbytes / args.shard_size) + 1


t_start = time.time()
shards = (ds.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))
filenames = (f"{args.out_path}/data/train-{index:05d}-of-{num_shards:05d}.parquet" for index in range(num_shards))

with Pool(16) as p:
    list(tqdm(p.imap_unordered(save_shard, zip(filenames, shards), chunksize=4), total=num_shards))
print(f"Time to save dataset: {time.time()-t_start:.2f}")

# To push to hub run `git add data/commit/push` inside dataset repo folder (the one cloned from HF: out_path/args.repo_name)
# no need to push duplicate_clusters.json