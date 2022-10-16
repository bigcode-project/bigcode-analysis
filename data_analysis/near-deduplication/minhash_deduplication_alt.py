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
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

multiprocessing.set_start_method("fork", force=True)

import networkit as nk
import numpy as np
import typer
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from datasketch import LeanMinHash, MinHash, MinHashLSH
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

random.seed(42)
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
# 4. https://docs.python.org/3/library/gc.html#gc.freeze

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
        # In practice, it might stuck here, you can hit Ctrl+C and run it again.
        for file in tqdm(sorted(glob.glob(conf["data_dir"] + "/*.jsonl")), desc="Loading datasets..."):
            datasets.append(load_dataset("json", data_files=file, split=conf["split"], cache_dir=conf["cache_dir"]))
        ds = concatenate_datasets(datasets)
        ds.save_to_disk(conf["concat_output"])
        ds = load_from_disk(conf["concat_output"])
    # Or load from the concatenated dataset
    else:
        ds = load_from_disk(conf["concat_output"])

    # Assign unique index to each record
    # A temporary filtering is used here
    ds = ds.filter(
        lambda x: len({t for t in NON_ALPHA.split(x[conf["column"]]) if t}) >= conf["min_token_length"],
        num_proc=os.cpu_count(),
        desc="Filtering records...",
    )
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
    Embed the content of a record into a MinHash object. This function should be
    used with multiprocessing and it scales well with the number of cores.

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
    >>> result = embed_func(0, "Hello world!", num_perm=128)
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
    Query the MinHashLSH index for the record. This function can be used with multiprocessing
    as long as the index is shared across processes.

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


def calculate_average_false_positive_rate(
    clusters: List[List[int]],
    reference_records: Iterable | Dataset,
    threshold: float,
    column: str,
):
    """
    Calculate the average false positive rate within each cluster. The false positives are defined as
    number of examples that have a maximum jaccard similarity with any example in the cluster that is
    less than the threshold. The false positive rate is defined as the number of false positives divided
    by the number of examples in the cluster. The average false positive rate is defined as the average
    of the false positive rate across all clusters given.

    Parameters
    ----------
    clusters : List[List[int]]
        The clusters of duplicate records.
    reference_records : Iterable | Dataset
        The reference records. It can be an iterable or a Dataset.
    threshold : float
        The threshold to use for calculating the false positive rate.
    column : str
        The column to use for calculating the false positive rate.
    """
    sample_size: int = 10
    cluster_false_positive_rates: List[float] = []
    deltas: List[float] = []

    for cluster in tqdm(clusters, desc="Calculating sampling false positive rate..."):
        num_false_positives = 0
        ids = sorted(cluster)
        for i, x in enumerate(ids):
            is_false_positive = True
            max_similarity = -float("inf")
            for j, y in enumerate(ids):
                if i == j:
                    continue
                # TODO This can be redundant but we only calculate this for a small sample
                similarity = jaccard_similarity(reference_records[x][column], reference_records[y][column])
                max_similarity = max(max_similarity, similarity)
                if max_similarity >= threshold:
                    is_false_positive = False
                    break
            if is_false_positive:
                num_false_positives += 1
                deltas.append(threshold - max_similarity)
        cluster_false_positive_rates.append(num_false_positives / len(ids))
        sample_size -= 1

    logger.info(
        f"Average false positive rate from {len(clusters)} clusters: {np.mean(cluster_false_positive_rates):.2f}"
    )
    logger.info(f"Average similarity delta from threshold: - {np.mean(deltas):.2f}")


def find_duplicate_communities(
    records: Iterable | Dataset,
    community_detection: bool,
    output: str,
    report_false_positive_rate: bool = False,
    reference_records: Iterable | Dataset | None = None,
    threashold: float = 0.85,
    column: str = "content",
    sample_size: int = 10,
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
    SAMPLE_MIN_SIZE = 10
    g = nk.graph.Graph()
    for record in tqdm(records, desc="Constructing graph..."):  # This creats a bottleneck since it is not parallelized.
        for y in record["__neighbors__"]:
            g.addEdge(record["__id__"], y, addMissing=True)

    to_remove: Set[int] = set()
    samples: List[List[int]] = []
    if not community_detection:
        cc = nk.components.ConnectedComponents(g)
        cc.run()
        partition = cc.getPartition()
        components = cc.getComponents()
        random.shuffle(components)
        for component in tqdm(components, desc="Iterating over components..."):
            component = sorted(component)
            to_remove.update(component[1:])
            if sample_size > 0 and len(component) > SAMPLE_MIN_SIZE:
                samples.append(component[:])
                sample_size -= 1
    else:
        partition = nk.community.detectCommunities(g)
        communities = partition.getSubsetIds()
        random.shuffle(communities)
        for i in tqdm(communities, desc="Iterating over communities..."):
            ids = partition.getMembers(i)
            to_remove.update(sorted(ids)[1:])
            if sample_size > 0 and len(ids) > SAMPLE_MIN_SIZE:
                samples.append(ids[:])
                sample_size -= 1

    nk.community.writeCommunities(partition, str(output))
    if report_false_positive_rate:
        calculate_average_false_positive_rate(
            samples,
            reference_records,
            threashold,
            column,
        )

    return to_remove


def jaccard_similarity(code1: str, code2: str) -> float:
    """
    Calculate the jaccard similarity between two code snippets.

    Parameters
    ----------
    code1 : str
        The first code snippet.
    code2 : str
        The second code snippet.
    
    Returns
    -------
    float
        The jaccard similarity between the two code snippets.

    Examples
    --------
    >>> jaccard_similarity("a = 1", "a = 2")
    0.3333333333333333
    >>> jaccard_similarity("a = 1", "a = 1")
    1.0
    """
    tokens1 = set([t for t in NON_ALPHA.split(code1) if t.strip()])
    tokens2 = set([t for t in NON_ALPHA.split(code2) if t.strip()])
    return len(tokens1 & tokens2) / max(1, len(tokens1 | tokens2))


def find_duplicate_non_extremes(
    records: Iterable | Dataset,
    reference_records: Iterable | Dataset,
    column: str,
    threshold: float,
) -> None:
    """
    This is a approximation of what has been used in other script.

    This is slow in this implementation as parallelization requires a global variable
    to hold the dataset to query, which is not implemented in this script.
    """
    g = nk.graph.Graph()
    for record in tqdm(records, desc="Constructing graph..."):
        for y in record["__neighbors__"]:
            g.addEdge(record["__id__"], y, addMissing=True)

    to_remove: Set[int] = set()
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    for component in tqdm(sorted(cc.getComponents(), key=len, reverse=True), desc="Iterating over components..."):
        extremes: Set[int] = set()
        # greedy clustering within each component
        for element1 in tqdm(component, leave=False):
            code1 = reference_records[element1][column]
            for element2 in extremes:
                code2 = reference_records[element2][column]
                if jaccard_similarity(code1, code2) >= threshold:
                    break
            else:
                extremes.add(element1)
        to_remove.update([i for i in component if i not in extremes])

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
        extremes: bool = typer.Option(
            False, "--extremes", "-e", help="Use `extremes` instead of community nor components"
        ),
        false_positive_rate: bool = typer.Option(
            False, "--false-positive-rate", "-f", help="Report false positive rate"
        ),
        min_token_length: int = typer.Option(10, help="Minimum token length"),
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
            "extremes": extremes,
            "false_positive_rate": false_positive_rate,
            "min_token_length": min_token_length,
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

            # This prevents the index's reference count being modified so it can be shared across processes
            # It will take some time as everything will be copied into a permanent memory space
            gc.disable()
            gc.freeze()

            # Sanity Check for copy-on-write fork
            # This is only a simple check. Python makes no guarantees about the id function and phsyical memory address.
            temp = embedded.select(range(os.cpu_count())).map(lambda _: {"index_address": id(lsh)}, num_proc=os.cpu_count(), remove_columns=embedded.column_names)
            assert len(set(temp["index_address"])) == 1, "Index is not shared across processes"

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

            gc.enable()
            gc.unfreeze()
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
            if not extremes:
                dup_ids = find_duplicate_communities(
                    queried,
                    conf["community_detection"],
                    conf["community_output"],
                    conf["false_positive_rate"],
                    ds,
                    conf["threshold"],
                    conf["column"],
                )
            else:
                dup_ids = find_duplicate_non_extremes(queried, ds, conf["column"], conf["threshold"])
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
        # ds = load_dataset_with_config(conf)
        final_data = ds.filter(
            lambda _, idx: idx not in dup_ids,
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )

        if "repository_name" in final_data.features and "path" in final_data.features:
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
