#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations
from collections import defaultdict

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

import typer
import graph_tool as gt
from graph_tool.all import label_components, minimize_nested_blockmodel_dl
from datasets import Dataset, Features, Value, load_dataset, load_from_disk
from datasketch import LeanMinHash
from datasketch import MinHash
from datasketch import MinHashLSH
from rich import box
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import track
from mpire import WorkerPool

"""
Find duplicates in a dataset using MinHash LSH.

1. Avoid calling ds[column]. This can be very slow and memory intensive.
2. Use ds.map to apply a function to the dataset.
3. Use WorkerPool to parallelize the computation if something needs to be shared across processes.
3. Store intermediate results in a temporary file to avoid recomputing.
"""

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))


def find_duplicate_communities(records: Iterable | Dataset, seed: int = 42) -> Set[int]:
    """
    Find the duplicate communities from the queried dataset.

    Parameters
    ----------
    records : Iterable | Dataset
        The dataset that contains both `__id__` and `__neighbors__`.
    seed : int, optional
        The seed for the random number generator, by default 42

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """

    # Original implementation: slow but more accurate
    # import networkx as nx
    # g = nx.Graph()
    # for record in track(records, description="Constructing graph..."):
    #     if not record["__neighbors__"]:
    #         continue
    #     g.add_node(record["__id__"])
    #     for y in record["__neighbors__"]:
    #         g.add_edge(record["__id__"], y)

    # to_remove: Set[int] = set()

    # for sub_graph in track(nx.connected_components(g), description="Finding duplicate communities..."):
    #     for c in nx.community.louvain_communities(g.subgraph(sub_graph), seed=seed):
    #         to_remove.update(sorted(c)[1:])
    
    # Remove all but one of the nodes in each connected component.
    ug = gt.Graph(directed=False)
    for record in track(records, description="Constructing graph..."):
        ug.add_edge_list([(record["__id__"], y) for y in record["__neighbors__"]])
    
    to_remove: Set[int] = set()
    cluster2ids = defaultdict(set)
    ug.properties[("v", "cluster")] = label_components(ug)[0]  # O(n)
    
    for idx, cluster in track(ug.get_vertices(vprops=[ug.vp.cluster]), description="Iterating over vertices..."):
        cluster2ids[cluster].add(idx)
    
    # This might cause false negatives, but it is much faster.
    for cluster, ids in track(cluster2ids.items(), description="Iterating over clusters..."):
        to_remove.update(sorted(ids)[1:])

    return to_remove


if __name__ == "__main__":

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
        from_neighbor_results: str = typer.Option(
            None, help="Resume from a dataset where neighbors have already been computed"
        ),
        to_neighbor_results: str = typer.Option(
            None, help="Store the neighbor results in a dataset, default f`{dataset}_{config}_{split}_{column}_neighbors`"
        ),
        from_duplicate_ids: str = typer.Option(
            None, help="Resume from a json file where duplicate ids have already been computed"
        ),
        to_duplicate_ids: str = typer.Option(
            None, help="Store the duplicate ids in a json file, default f`{dataset}_{config}_{split}_{column}_duplicate_ids.json`"
        ),
        output: str = typer.Option("results", help="Store the deduplicated data in a dataset"),
    ):
        if to_neighbor_results is None:
            to_neighbor_results = f"{dataset}_{config}_{split}_{column}_neighbors"
        if to_duplicate_ids is None:
            to_duplicate_ids = f"{dataset}_{config}_{split}_{column}_duplicate_ids.json"
        
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
            "from_neighbor_results": from_neighbor_results,
            "to_neighbor_results": to_neighbor_results,
            "from_duplicate_ids": from_duplicate_ids,
            "to_duplicate_ids": to_duplicate_ids,
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

        def query_func(index: MinHashLSH, **record: Dict[str, Any]) -> Dict[str, Any]:
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

        N = len(split_data)
        start_time = time.time()

        if not from_neighbor_results:

            embedded = split_data.map(
                function=embed_func,
                num_proc=os.cpu_count(),
                with_indices=True,
                desc=f"Fingerprinting...",
                remove_columns=split_data.column_names,
            )

            del split_data

            with lsh.insertion_session() as session:
                for data in track(embedded, description="Indexing signatures"):
                    if data["__id__"] in lsh.keys:
                        continue
                    session.insert(
                        data["__id__"],
                        LeanMinHash(seed=conf["seed"], hashvalues=data["__signature__"]),
                        check_duplication=False,  # We have already checked for duplicates.
                    )

            # As multiprocessing copies the index, we avoid this if necessary.
            # queried = embedded.map(
            #     function=lambda x: query_func(lsh, x),
            #     num_proc=os.cpu_count(),
            #     desc=f"Querying...",
            #     # providing this seems to unstuck the hashing process
            #     new_fingerprint=hashlib.md5(json.dumps(conf).encode()).hexdigest(),
            #     remove_columns=embedded.column_names,
            # )

            # This reduces the memory load but it is slower and not cached
            with WorkerPool(n_jobs=os.cpu_count(), shared_objects=lsh) as pool:
                queried = pool.map(
                    query_func,
                    embedded,
                    progress_bar=True,
                    progress_bar_options={"desc": "Querying..."},
                )
                queried = Dataset.from_list(
                    queried, features=Features({"__neighbors__": [Value("int64")], "__id__": Value("int64")})
                )

            # if conf["verbose"]:
            #     # print some examples
            #     duplicates = queried.filter(
            #         lambda x: len(x["__neighbors__"]) > 0, num_proc=os.cpu_count(), desc="Finding duplicates..."
            #     )

            #     table = Table(
            #         title="Some examples of duplicate code",
            #         show_header=True,
            #         header_style="bold magenta",
            #         box=box.HORIZONTALS,
            #     )
            #     table.add_column("id", style="dim", width=12)
            #     table.add_column("dup id", style="dim", width=12)
            #     table.add_column("code", width=80)
            #     table.add_column("dup code", width=80)

            #     for i in range(10):
            #         curr_id = duplicates[i]["__id__"]
            #         curr_code = split_data[curr_id][conf["column"]]
            #         for dup_id in duplicates[i]["__neighbors__"][:3]:
            #             table.add_row(
            #                 str(curr_id),
            #                 str(dup_id),
            #                 "\n".join(textwrap.wrap(curr_code[:240], width=80, placeholder="...")),
            #                 "\n".join(
            #                     textwrap.wrap(split_data[dup_id][conf["column"]][:240], width=80, placeholder="...")
            #                 ),
            #             )

            #         table.add_row(end_section=True)

            #     console.print(table)

            queried.save_to_disk(to_neighbor_results)

        else:

            queried = load_from_disk(from_neighbor_results)
        
        del lsh


        if not from_duplicate_ids:
            queried = queried.filter(
                lambda x: len(x["__neighbors__"]) > 0,
                num_proc=os.cpu_count(),
                desc="Finding duplicates..."
            )
            dup_ids = find_duplicate_communities(queried, seed=conf["seed"])

            with open(to_duplicate_ids, "w") as f:
                json.dump(list(map(str, dup_ids)), f)
        else:
            with open(from_duplicate_ids, "r") as f:
                dup_ids = set((map(int, json.load(f))))

        del queried

        sys.stderr.flush()
        sys.stdout.flush()
        print()

        logger.info(f"Original size: {N}")
        logger.info(f"Removed size: {len(dup_ids)} ({len(dup_ids) / N* 100:.2f}%)")
        logger.info(
            f"Final size: {N - len(dup_ids)} ({(N - len(dup_ids)) / N * 100:.2f}%)"
        )
        logger.info(f"Processing time taken: {time.time() - start_time:.2f} seconds")

        split_data = load_dataset(
            conf["dataset"],
            conf["config"],
            split=conf["split"],
            use_auth_token=True,
            cache_dir=conf["cache_dir"],
        )

        final_data = split_data.filter(
            lambda _, idx: idx not in dup_ids,  # this will copy dup_ids to each process
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering duplicates...",
        )
        final_data.save_to_disk(output)

        logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
        logger.info("🤗 Happy Deduplicating 🤗")

    typer.run(run)
