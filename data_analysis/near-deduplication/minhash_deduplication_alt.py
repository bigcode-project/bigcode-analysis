#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import gc
import glob
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Set

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
multiprocessing.set_start_method("fork", force=True)

import networkit as nk
import numpy as np
import typer
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasketch import LeanMinHash, MinHash, MinHashLSH
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

MINHASH_SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))

# With multiprocessing and copy-on-write fork (Linux and macOS), 
# we can use global variables to share objects across processes.
# This might not be the case on some systems where objects are 
# pickled and sent to the child processes. It might also not be reflected 
# when you use top command to check the memory usage. One way to check is to 
# print the id of the object in the child processes and see if they are the same.
# References: 
# 1. https://stackoverflow.com/questions/38084401/leveraging-copy-on-write-to-copy-data-to-multiprocessing-pool-worker-process
# 2. https://stackoverflow.com/questions/53841599/python-multiprocessing-copy-on-write-behaving-differently-between-osx-and-ubuntu
# 3. https://stackoverflow.com/questions/40221868/multiprocessing-global-variable-memory-copying

lsh: MinHashLSH | None = None
dup_ids: Set[int] | None = None


def load_dataset_with_config(conf: Dict[str, Any]) -> Dataset:
    """
    Load a dataset based on the configuration. Be careful about changing this function,
    as it is used for caching the intermediate results.

    Parameters
    ----------
    conf : Dict[str, Any]
        The configuration. Mainly, there are three ways to load a dataset:
        1. Directly from th ehub
        2. From a local git repository
        3. From a local dataset directory that was saved by `save_to_disk` before
    
    Returns
    -------
    Dataset
        The loaded dataset.
    """

    # Load from hub
    if not conf["lfs"]:
        ds = load_dataset(
            conf["dataset"],
            conf["config"],
            data_dir=conf["data_dir"],
            split=conf["split"],
            use_auth_token=True,
            cache_dir=conf["cache_dir"],
        )
    # Or load from git lfs files
    elif not os.path.exists(conf["concat_output"]):
        datasets = []
        for file in tqdm(sorted(glob.glob(conf["data_dir"] + "/*.jsonl")), desc="Loading datasets..."):
            datasets.append(load_dataset("json", data_files=file, split=conf["split"], cache_dir=conf["cache_dir"]))
        ds = concatenate_datasets(datasets)
        ds.save_to_disk(conf["concat_output"])
        ds = load_from_disk(conf["concat_output"])
    # Or load from the concatenated dataset
    else:
        ds = load_from_disk(conf["concat_output"])

    # Assign unique index to each record
    ds = ds.map(
        lambda _, idx: {"__id__": idx},
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Adding index...",
    )

    if conf["sample_size"] > 0:
        ds = ds.select(range(conf["sample_size"]))

    return ds


def embed_func(idx: int, content: str, *, num_perm: int) -> Dict[str, Any]:
    """
    Embed the content of a record into a MinHash object.

    Parameters
    ----------
    idx : int
        The index of the record.
    content : str
        The content to embed.
    num_perm : int
        The number of permutations to use in the MinHash object.
    seed : int
        The seed to use in the MinHash object.

    Returns
    -------
    Dict[str, Any]
        The MinHash signature and the index of the record.

    Examples
    --------
    >>> result = embed_func(0, "Hello world!", num_perm=128, seed=42)
    >>> result["__id__"]
    0
    >>> result["__signature__"].shape
    (128,)
    >>> result["__signature__"].dtype
    dtype('uint64')
    """
    m = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
    m.update_batch([token.encode("utf-8") for token in {t for t in NON_ALPHA.split(content) if t}])
    return {"__signature__": m.hashvalues, "__id__": idx}


def query_func(idx: int, signature: np.ndarray, *, index: MinHashLSH) -> Dict[str, Any]:
    """
    Query the MinHashLSH index for the record.

    Parameters
    ----------
    index : MinHashLSH
        The MinHashLSH index. It is shared across all processes when using multiprocessing with fork without copy.
    record : Dict[str, Any]
        The record to query.

    Returns
    -------
    Dict[str, Any]
        The query result.
    """
    return {
        "__neighbors__": [
            dup_idx
            for dup_idx in index.query(
                LeanMinHash(seed=MINHASH_SEED, hashvalues=signature),
            )
            if dup_idx != idx  # exclude itself
        ],
        "__id__": idx,
    }


def find_duplicate_communities(
    records: Iterable | Dataset, 
    community_detection: bool,
    output: str
) -> Set[int]:
    """
    Find the duplicate communities from the queried dataset.

    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains both `__id__` and `__neighbors__`.
    community_detection : bool
        Whether to use community detection to find the duplicate communities, or to use the connected components.
    output : str
        The output file to save the duplicate communities.

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """
    g = nk.graph.Graph()
    for record in tqdm(records, desc="Constructing graph..."):  # This creats a bottleneck since it is not parallelized.
        for y in record["__neighbors__"]:
            g.addEdge(record["__id__"], y, addMissing=True)

    to_remove: Set[int] = set()
    if not community_detection:
        cc = nk.components.ConnectedComponents(g)
        cc.run()
        partition = cc.getPartition()
        for component in tqdm(cc.getComponents(), desc="Iterating over components..."):
            to_remove.update(sorted(component)[1:])
    else:
        partition = nk.community.detectCommunities(g)
        for i in tqdm(partition.getSubsetIds(), desc="Iterating over communities..."):
            ids = partition.getMembers(i)
            to_remove.update(sorted(ids)[1:])
    nk.community.writeCommunities(partition, str(output))


    return to_remove


if __name__ == "__main__":

    def run(
        dataset: str = typer.Option("codeparrot/codeparrot-clean-valid", help="The dataset to use"),
        config: str = typer.Option("default", help="Dataset config"),
        data_dir: str = typer.Option(None, help="Dataset data directory"),
        split: str = typer.Option("train", help="Dataset split"),
        column: str = typer.Option("content", help="Dataset column"),
        cache_dir: str = typer.Option(".cache", help="Cache directory"),
        num_perm: int = typer.Option(256, help="Number of permutations"),
        seed: int = typer.Option(42, help="Random seed"),
        threshold: float = typer.Option(0.85, help="Minhash threshold"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        sample_size: int = typer.Option(-1, help="Sample size"),
        input_neighbor_dataset: str = typer.Option(None, help="Resume from a queried dataset"),
        output_neighbor_dataset: str = typer.Option(None, help="Store a queried dataset"),
        input_duplicate_ids: str = typer.Option(None, help="Resume from computed duplicate ids"),
        output_duplicate_ids: str = typer.Option(None, help="Store computed duplicate ids"),
        output: str = typer.Option(None, help="Store the deduplicated dataset"),
        lfs: bool = typer.Option(False, help="Use LFS files"),
        community_detection: bool = typer.Option(False, "--community-detection", "-c", help="Use community detection"),
    ):
        global lsh
        global dup_ids

        OUTPUT_BASE = Path("results") / dataset / config / (data_dir or "all") / split / column
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

        output_neighbor_dataset = output_neighbor_dataset or (OUTPUT_BASE / "neighbors")
        output_duplicate_ids = output_duplicate_ids or (OUTPUT_BASE / "duplicate_ids.json")
        output = output or (OUTPUT_BASE / "deduplicated")
        output_concat = OUTPUT_BASE / "concat"
        output_index = OUTPUT_BASE / "index.pkl"
        output_unique_paths = OUTPUT_BASE / "unique_paths.json"
        output_community = OUTPUT_BASE / "community.partition"

        logger.info(f"Output base: {OUTPUT_BASE}")
        logger.info(f"Concat output: {output_concat}")
        logger.info(f"Index output: {output_index}")
        logger.info(f"Neighbor dataset output: {output_neighbor_dataset}")
        logger.info(f"Duplicate ids output: {output_duplicate_ids}")
        logger.info(f"Unique paths output: {output_unique_paths}")
        logger.info(f"Community output: {output_community}")
        logger.info(f"Output: {output}")

        conf = {
            "cache_dir": cache_dir,
            "num_perm": num_perm,
            "seed": seed,
            "threshold": threshold,
            "dataset": dataset,
            "config": config,
            "data_dir": data_dir,
            "split": split,
            "column": column,
            "verbose": verbose,
            "sample_size": sample_size,
            "input_neighbor_dataset": input_neighbor_dataset,
            "input_neighbor_dataset": input_neighbor_dataset,
            "input_duplicate_ids": input_duplicate_ids,
            "output_duplicate_ids": output_duplicate_ids,
            "output": output,
            "lfs": lfs,
            "index_output": output_index,
            "concat_output": output_concat,
            "community_detection": community_detection,
            "community_output": output_community,
        }

        lsh = MinHashLSH(
            threshold=conf["threshold"],
            num_perm=conf["num_perm"],
        )

        ds = load_dataset_with_config(conf)
        DATA_SIZE = len(ds)
        start_time = time.time()

        if not input_neighbor_dataset and not input_duplicate_ids:

            # region: embed
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={"num_perm": conf["num_perm"]},
                input_columns=["__id__", conf["column"]],
                remove_columns=[conf["column"]],
                num_proc=os.cpu_count(),
                desc=f"Fingerprinting...",
            )

            del ds
            gc.collect()
            # endregion

            # region: index
            if os.path.exists(output_index):
                logger.info(f"Loading index from {output_index}")
                with open(output_index, "rb") as f:
                    lsh = pickle.load(f)
            else:
                with lsh.insertion_session() as session:
                    for data in tqdm(embedded, desc="Indexing signatures..."):
                        if data["__id__"] in lsh:
                            continue
                        session.insert(
                            data["__id__"],
                            LeanMinHash(seed=MINHASH_SEED, hashvalues=data["__signature__"]),
                            check_duplication=False,
                        )
                pickle.dump(lsh, open(output_index, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            # endregion

            # region: query
            queried = embedded.map(
                lambda x, y: query_func(x, y, index=lsh),  # Do not use fn_kwargs here, it will be pickled instead.
                num_proc=os.cpu_count(),
                new_fingerprint=hashlib.md5(pickle.dumps(conf)).hexdigest(),
                input_columns=["__id__", "__signature__"],
                remove_columns=["__signature__"],
                desc=f"Querying...",
            )
            # endregion

            queried.save_to_disk(output_neighbor_dataset)
        elif not input_duplicate_ids:
            queried = load_from_disk(input_neighbor_dataset)
        else:
            queried = None

        del lsh
        gc.collect()

        if not input_duplicate_ids:
            queried = queried.filter(
                lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
            )
            dup_ids = find_duplicate_communities(queried, conf["community_detection"], conf["community_output"])

            with open(output_duplicate_ids, "w") as f:
                json.dump(list(map(str, dup_ids)), f)
        else:
            with open(input_duplicate_ids, "r") as f:
                dup_ids = set((map(int, json.load(f))))

        del queried
        gc.collect()

        logger.info(f"Processing time taken: {time.time() - start_time:.2f} seconds")

        # region: deduplicate
        # Reload the original dataset
        ds = load_dataset_with_config(conf)
        final_data = ds.filter(
            lambda _, idx: idx not in dup_ids,
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )

        with open(output_unique_paths, "w") as f:
            temp = final_data.map(
                lambda x: {"url": x["repository_name"] + "/" + x["path"]},
                num_proc=os.cpu_count(),
                remove_columns=final_data.column_names,
                desc="Saving unique paths...",
            )
            json.dump(list(set(temp["url"])), f)

        final_data.save_to_disk(output)
        # endregion

        FINAL_DATA_SIZE = len(final_data)
        DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
        LAN = (data_dir or "all").split("/")[-1]
        logger.info(
            f"| {LAN} "
            f"| {DATA_SIZE} "
            f"| {DUP_SIZE} ({DUP_SIZE / DATA_SIZE * 100:.2f}%) "
            f"| {time.time() - start_time:.2f} sec |"
        )
        logger.info("🤗 Happy Deduplicating 🤗")

    typer.run(run)
