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
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import networkx as nx
from datasets import load_dataset
from datasketch import LeanMinHash
from datasketch import MinHash
from datasketch import MinHashLSH
from rich import box
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from tqdm import tqdm

NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))


def find_duplicate_communities(pairs: List[Tuple[int, List[int]]] | Iterable) -> Set[int]:
    """
    Find the duplicate communities from pairs of (id, duplicate_ids).

    Parameters
    ----------
    pairs : List[Tuple[int, List[int]]] | Iterable
        The list of (id, duplicate_ids) pairs.

    Returns
    -------
    Set[int]
        The set of duplicate ids that should be removed, leaving only one id in each community.
    """

    g = nx.Graph()
    for x, neighbors in tqdm(pairs, desc="Constructing graph..."):
        g.add_node(x)
        for y in neighbors:
            g.add_edge(x, y)

    to_remove: Set[int] = set()

    for c in tqdm(nx.community.louvain_communities(g), desc="Finding communities..."):
        to_remove.update(sorted(c)[1:])

    return to_remove


if __name__ == "__main__":

    conf = {
        "cache_dir": ".cache",
        "num_perm": 256,
        "seed": 42,
        "threshold": 0.85,
        "dataset": "codeparrot/codeparrot-clean-valid",
        "config": "default",
        "split": "train",
        "column": "content",
        "verbose": True,
    }


    def embed_func(record, idx):
        m = MinHash(num_perm=conf["num_perm"], seed=conf["seed"])
        m.update_batch(
            [token.encode("utf-8") for token in set([t for t in NON_ALPHA.split(record[conf["column"]]) if t])]
        )
        return {"__signature__": m.hashvalues, "__id__": idx}


    def query_func(index, record, idx):
        return {
            "__neighbors__": [
                dup_idx
                for x in index.query(
                    LeanMinHash(seed=conf["seed"], hashvalues=record["__signature__"]),
                )
                if (dup_idx := int(x.split("-", 1)[1])) != idx
            ]
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

    start_time = time.time()

    split_data = split_data.map(
        function=embed_func,
        num_proc=os.cpu_count(),
        with_indices=True,
        desc=f"Fingerprinting...",
    )

    with lsh.insertion_session() as session:
        for key, data in enumerate(tqdm(split_data, desc="Indexing signatures")):
            if f"id-{key}" in lsh.keys:
                continue
            session.insert(
                f"id-{key}",
                LeanMinHash(seed=conf["seed"], hashvalues=data["__signature__"]),
            )

    split_data = split_data.map(
        function=lambda x, idx: query_func(lsh, x, idx),
        num_proc=os.cpu_count(),
        desc=f"Querying...",
        with_indices=True,
        new_fingerprint=hashlib.md5(json.dumps(conf).encode()).hexdigest(),  # this is needed to skip hasher
    )

    if conf["verbose"]:
        # print some examples
        duplicates = split_data.filter(
            lambda x: len(x["__neighbors__"]) > 0,
            num_proc=os.cpu_count(),
            desc="Finding duplicates..."
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
            curr_code = duplicates[i][conf["column"]]
            for dup_id in duplicates[i]["__neighbors__"][:3]:
                table.add_row(
                    str(curr_id),
                    str(dup_id),
                    "\n".join(textwrap.wrap(curr_code[:240], width=80, placeholder="...")),
                    "\n".join(textwrap.wrap(split_data[dup_id][conf["column"]][:240], width=80, placeholder="...")),
                )

            table.add_row(end_section=True)

        console.print(table)

    dup_ids = find_duplicate_communities(zip(range(len(split_data)), split_data["__neighbors__"]))

    sys.stderr.flush()
    sys.stdout.flush()
    print()

    logger.info(f"Original size: {len(split_data)}")
    logger.info(f"Removed size: {len(dup_ids)}")
    logger.info(f"Processing time taken: {time.time() - start_time:.2f} seconds")

    final_data = split_data.filter(
        lambda x, idx: idx not in dup_ids,
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Filtering duplicates..."
    )
    final_data = final_data.remove_columns(["__signature__", "__neighbors__", "__id__"])
    final_data.save_to_disk("results")

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
    logger.info("🤗 Happy Deduplicating 🤗")
