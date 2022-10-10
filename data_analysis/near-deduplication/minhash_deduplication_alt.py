#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Set

import graph_tool as gt
import typer
from datasets import Dataset, load_dataset, load_from_disk
from datasketch import LeanMinHash, MinHash, MinHashLSH
from graph_tool.all import label_components
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))


def load_dataset_with_config(conf: Dict[str, Any]):

    ds = load_dataset(
        conf["dataset"],
        conf["config"],
        data_dir=conf["data_dir"],
        split=conf["split"],
        use_auth_token=True,
        cache_dir=conf["cache_dir"],
    )
    ds = ds.map(
        lambda _, idx: {"__id__": idx},
        with_indices=True,
        num_proc=os.cpu_count(),
    )

    if conf["sample_size"] > 0:
        ds = ds.select(range(conf["sample_size"]))

    return ds


def embed_func(idx: int, content: str, *, num_perm: int, seed: int) -> Dict[str, Any]:
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
    m = MinHash(num_perm=num_perm, seed=seed)
    m.update_batch([token.encode("utf-8") for token in {t for t in NON_ALPHA.split(content) if t}])
    return {"__signature__": m.hashvalues, "__id__": idx}


def query_func(idx: int, signature, *, index: MinHashLSH, seed: int) -> Dict[str, Any]:
    """
    Query the MinHashLSH index for the record.

    Parameters
    ----------
    signature
        The signature of the record.
    idx : int
        The index of the record.
    index : MinHashLSH
        The MinHashLSH index. It is shared across all processes when using multiprocessing with fork without copy.
    seed : int
        The seed to use in the MinHash object.

    Returns
    -------
    Dict[str, Any]
        The query result.
    """
    return {
        "__neighbors__": [
            dup_idx
            for dup_idx in index.query(
                LeanMinHash(seed=seed, hashvalues=signature),
            )
            if dup_idx != idx  # exclude self
        ],
        "__id__": idx,
    }


def find_duplicate_communities(records: Iterable | Dataset, community_detection: bool = False) -> Set[int]:
    """
    Find the duplicate communities from the queried dataset.

    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains both `__id__` and `__neighbors__`.
    community_detection : bool
        Whether to use community detection to find the duplicate communities, or to use the connected components.

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """
    # Remove all but one of the nodes in each connected component.
    # This might cause false negatives, but it is much faster than finding communities.
    assert community_detection is False, "Community detection is not implemented yet."
    ug = gt.Graph(directed=False)
    for record in tqdm(records, desc="Constructing graph..."):
        ug.add_edge_list([(record["__id__"], y) for y in record["__neighbors__"]])

    to_remove: Set[int] = set()
    cluster2ids = defaultdict(set)
    ug.properties[("v", "cluster")] = label_components(ug)[0]  # O(n)

    for idx, cluster in tqdm(ug.get_vertices(vprops=[ug.vp.cluster]), desc="Iterating over vertices..."):
        cluster2ids[cluster].add(idx)

    for cluster, ids in tqdm(cluster2ids.items(), desc="Iterating over clusters..."):
        to_remove.update(sorted(ids)[1:])

    return to_remove


if __name__ == "__main__":

    def run(
        dataset: str = typer.Option("codeparrot/codeparrot-clean-valid", help="The dataset to use"),
        config: str = typer.Option("default", help="Dataset config"),
        data_dir: str = typer.Option(None, help="Dataset data directory"),
        split: str = typer.Option("train", help="Dataset split"),
        column: str = typer.Option("content", help="Dataset column"),
        cache_dir: str = typer.Option(".cache", help="Cache directory"),
        num_perm: int = typer.Option(128, help="Number of permutations"),
        seed: int = typer.Option(42, help="Random seed"),
        threshold: float = typer.Option(0.85, help="Minhash threshold"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        sample_size: int = typer.Option(-1, help="Sample size"),
        input_neighbor_dataset: str = typer.Option(None, help="Resume from a queried dataset"),
        output_neighbor_dataset: str = typer.Option(None, help="Store a queried dataset"),
        input_duplicate_ids: str = typer.Option(None, help="Resume from computed duplicate ids"),
        output_duplicate_ids: str = typer.Option(None, help="Store computed duplicate ids"),
        output: str = typer.Option(None, help="Store the deduplicated dataset"),
    ):

        OUTPUT_BASE = Path("results") / dataset / config / (data_dir or "all") / split / column
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

        output_neighbor_dataset = output_neighbor_dataset or OUTPUT_BASE / "neighbors"
        output_duplicate_ids = output_duplicate_ids or OUTPUT_BASE / "duplicate_ids.json"
        output = output or OUTPUT_BASE / "deduplicated"

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
        }

        # Use a database backend so that the index can be shared across all processes.
        lsh = MinHashLSH(
            threshold=conf["threshold"],
            num_perm=conf["num_perm"],
            storage_config={
                "type": "redis",
                "redis": {"host": "localhost", "port": 6379},
            },
        )

        ds = load_dataset_with_config(conf)
        DATA_SIZE = len(ds)
        start_time = time.time()

        if not input_neighbor_dataset:

            # region: embed
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={"num_perm": conf["num_perm"], "seed": conf["seed"]},
                input_columns=["__id__", conf["column"]],
                remove_columns=[conf["column"]],
                num_proc=os.cpu_count(),
                desc=f"Fingerprinting...",
            )

            del ds
            gc.collect()
            # endregion

            # region: index
            with lsh.insertion_session(buffer_size=100000) as session:
                for data in tqdm(embedded, desc="Indexing signatures..."):
                    session.insert(
                        data["__id__"],
                        LeanMinHash(seed=conf["seed"], hashvalues=data["__signature__"]),
                        check_duplication=False,  # We assigned unique ids, so we skip this check.
                    )
            # endregion

            # region: query
            queried = embedded.map(
                function=query_func,
                fn_kwargs={"index": lsh, "seed": conf["seed"]},
                input_columns=["__id__", "__signature__"],
                remove_columns=["__signature__"],
                num_proc=os.cpu_count(),
                desc=f"Querying...",
            )
            # endregion

            queried.save_to_disk(output_neighbor_dataset)
        else:
            queried = load_from_disk(input_neighbor_dataset)

        del lsh
        gc.collect()

        if not input_duplicate_ids:
            queried = queried.filter(
                lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
            )
            dup_ids = find_duplicate_communities(queried)

            with open(output_duplicate_ids, "w") as f:
                json.dump(list(map(str, dup_ids)), f)
        else:
            with open(input_duplicate_ids, "r") as f:
                dup_ids = set((map(int, json.load(f))))

        del queried
        gc.collect()

        logger.info(f"Processing time taken: {time.time() - start_time:.2f} seconds")

        # region: deduplicate
        ds = load_dataset_with_config(conf)
        final_data = ds.filter(
            lambda _, idx: idx not in dup_ids,  # this will copy dup_ids to each process when using multiprocessing
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )

        del ds
        gc.collect()

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
