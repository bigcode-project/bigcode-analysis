#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, Iterable
from typing import Set

import networkx as nx
from datasets import Dataset, load_dataset, load_from_disk
from datasketch import LeanMinHash
from datasketch import MinHash
from datasketch import MinHashLSH
from rich import box
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from tqdm import tqdm

"""
Find duplicates in a dataset using MinHash LSH.

1. Avoid calling ds[column]. This can be very slow and memory intensive.
2. Use ds.map to apply a function to the dataset.
3. Store intermediate results in a temporary file to avoid recomputing.
4. Use global variables to be shared across processes when not modifying.
"""

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))


def find_duplicate_communities(records: Iterable | Dataset, seed: int = 42) -> Set[int]:
    """
    Find the duplicate communities from pairs of (id, duplicate_ids).

    Parameters
    ----------
    records: Iterable | Dataset
        The dataset.
    seed : int, optional
        The seed for the random number generator, by default 42

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """

    g = nx.Graph()
    for record in tqdm(records, desc="Constructing graph..."):
        if not record["__neighbors__"]:
            continue
        g.add_node(record["__id__"])
        for y in record["__neighbors__"]:
            g.add_edge(record["__id__"], y)

    to_remove: Set[int] = set()

    for sub_graph in tqdm(nx.connected_components(g), desc="Finding duplicate communities..."):
        for c in nx.community.louvain_communities(g.subgraph(sub_graph), seed=seed):
            to_remove.update(sorted(c)[1:])

    return to_remove


if __name__ == "__main__":

    import typer

    # import tracemalloc

    # Make sure the index is global so it can be shared across processes
    lsh = None
    dup_ids = set()

    def run(
        dataset: str = typer.Option(
            "codeparrot/codeparrot-clean-valid", help="The dataset to run the deduplication on"
        ),
        config: str = typer.Option("default", help="The config to use for the dataset"),
        split: str = typer.Option("train", help="The split to use for the dataset"),
        column: str = typer.Option("content", help="The column to use for the dataset"),
        cache_dir: str = typer.Option(".cache", help="Cache directory for datasets"),
        num_perm: int = typer.Option(256, help="Number of permutations for MinHash"),
        seed: int = typer.Option(42, help="Seed for random number generator"),
        threshold: float = typer.Option(0.85, help="Threshold for MinHash"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        sample_size: int = typer.Option(-1, help="Sample size for the dataset"),
        from_intermediate_results: str = typer.Option(None, help="Resume from intermediate dataset where neighbors are stored"),
        to_intermediate_results: str = typer.Option(
            "neighbors", help="Intermediate dataset where neighbors will be stored"
        ),
    ):
        global lsh
        global dup_ids

        conf = {
            "cache_dir": cache_dir,
            "num_perm": num_perm,
            "seed": seed,
            "threshold": threshold,
            "dataset": dataset,
            "config": config,
            "split": split,
            "column": column,
            "verbose": verbose,
            "sample_size": sample_size,
        }

        def embed_func(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
            """
            Embed the content of a record into a MinHash object.

            Parameters
            ----------
            record : Dict[str, Any]
                The record to embed.
            idx : int
                The index of the record.

            Returns
            -------
            Dict[str, Any]
                The MinHash signature and the index of the record.
            """
            m = MinHash(num_perm=conf["num_perm"], seed=conf["seed"])
            m.update_batch(
                [token.encode("utf-8") for token in {t for t in NON_ALPHA.split(record[conf["column"]]) if t}]
            )
            return {"__signature__": m.hashvalues, "__id__": idx}

        def query_func(index: MinHashLSH, record: Dict[str, Any]) -> Dict[str, Any]:
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
                        LeanMinHash(seed=conf["seed"], hashvalues=record["__signature__"]),
                    )
                    if dup_idx != record["__id__"]  # exclude self
                ],
                "__id__": record["__id__"],
            }

        lsh = MinHashLSH(
            threshold=conf["threshold"],
            num_perm=conf["num_perm"],
        )

        split_data = load_dataset(
            conf["dataset"],
            conf["config"],
            split=conf["split"],
            use_auth_token=True,
            cache_dir=conf["cache_dir"],
        )

        if conf["sample_size"] > 0:
            split_data = split_data.select(range(conf["sample_size"]))

        start_time = time.time()

        if not from_intermediate_results:

            embedded = split_data.map(
                function=embed_func,
                num_proc=os.cpu_count(),
                with_indices=True,
                desc=f"Fingerprinting...",
                remove_columns=split_data.column_names,
            )

            with lsh.insertion_session() as session:
                for data in tqdm(embedded, desc="Indexing signatures"):
                    if data["__id__"] in lsh.keys:
                        continue
                    session.insert(
                        data["__id__"],
                        LeanMinHash(seed=conf["seed"], hashvalues=data["__signature__"]),
                        check_duplication=False,  # We have already checked for duplicates.
                    )

            queried = embedded.map(
                function=lambda x: query_func(lsh, x),
                num_proc=os.cpu_count(),
                desc=f"Querying...",
                # providing this seems to unstuck the hashing process
                new_fingerprint=hashlib.md5(json.dumps(conf).encode()).hexdigest(),
                remove_columns=embedded.column_names,
            )

            if conf["verbose"]:
                # print some examples
                duplicates = queried.filter(
                    lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
                )

                table = Table(
                    title="Some examples of duplicate code",
                    show_header=True,
                    header_style="bold magenta",
                    box=box.HORIZONTALS,
                )
                table.add_column("id", style="dim", width=12)
                table.add_column("dup id", style="dim", width=12)
                table.add_column("code", width=80)
                table.add_column("dup code", width=80)

                for i in range(10):
                    curr_id = duplicates[i]["__id__"]
                    curr_code = split_data[curr_id][conf["column"]]
                    for dup_id in duplicates[i]["__neighbors__"][:3]:
                        table.add_row(
                            str(curr_id),
                            str(dup_id),
                            "\n".join(textwrap.wrap(curr_code[:240], width=80, placeholder="...")),
                            "\n".join(
                                textwrap.wrap(split_data[dup_id][conf["column"]][:240], width=80, placeholder="...")
                            ),
                        )

                    table.add_row(end_section=True)

                console.print(table)

            queried.save_to_disk(to_intermediate_results)

        else:

            queried = load_from_disk(from_intermediate_results)

        dup_ids = find_duplicate_communities(queried, seed=conf["seed"])

        sys.stderr.flush()
        sys.stdout.flush()
        print()

        logger.info(f"Original size: {len(split_data)}")
        logger.info(f"Removed size: {len(dup_ids)} ({len(dup_ids) / len(split_data) * 100:.2f}%)")
        logger.info(
            f"Final size: {len(split_data) - len(dup_ids)} ({(len(split_data) - len(dup_ids)) / len(split_data) * 100:.2f}%)"
        )
        logger.info(f"Processing time taken: {time.time() - start_time:.2f} seconds")

        final_data = split_data.filter(
            lambda _, idx: idx not in dup_ids,
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )
        final_data.save_to_disk("results")

        logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
        logger.info("🤗 Happy Deduplicating 🤗")

    # Using tracemalloc slows down the process by a lot, so we only use it when needed.
    # tracemalloc.start()
    typer.run(run)
    # used = tracemalloc.get_tracemalloc_memory()
    # logger.info(f"Memory used: {used / 1024 / 1024:.2f} MB")
    # tracemalloc.stop()
