#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import logging
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

import datasets
import typer
from datasets import load_dataset
from datasketch import MinHash
from datasketch import MinHashLSH
from nltk.util import ngrams
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
logger.propagate = False
datasets.logging.set_verbosity_error()


def embed_func(
    content: str, idx: int, *, num_perm: int, ngram_size: int, hashranges: List[Tuple[int, int]]
) -> Dict[str, Any]:
    m = MinHash(num_perm=num_perm, seed=MINHASH_SEED)
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    m.update_batch([token.encode("utf-8") for token in tokens])
    Hs = [bytes(m.hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


if __name__ == "__main__":

    def run(
        dataset: str = typer.Option("codeparrot/codeparrot-clean-valid", help="The dataset to use"),
        config: str = typer.Option("default", help="Dataset config"),
        split: str = typer.Option("train", help="Dataset split"),
        data_dir: str = typer.Option(None, help="Dataset data directory"),
        revision: str = typer.Option("main", help="Dataset revision"),
        column: str = typer.Option("content", help="Dataset column"),
        cache_dir: str = typer.Option(".cache", help="Cache directory"),
        ngram_size: int = typer.Option(5, help="The ngram size to use for MinHash"),
        num_perm: int = typer.Option(256, help="Number of permutations"),
        threshold: float = typer.Option(0.85, help="Minhash threshold"),
        output: str = typer.Option(None, help="Store the deduplicated dataset"),
    ):
        OUTPUT_BASE = Path(output or "output")
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
        output = OUTPUT_BASE / "deduplicated"
        time_measures = {}
        start_time = time.time()
        lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
        )

        time_measures["load_dataset"] = time.time()
        ds = load_dataset(
            dataset,
            config,
            data_dir=data_dir,
            split=split,
            use_auth_token=True,
            cache_dir=cache_dir,
            revision=revision,
            num_proc=os.cpu_count(),
        )
        time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
        DATA_SIZE = len(ds)

        time_measures["minhash"] = time.time()
        embedded = ds.map(
            function=embed_func,
            fn_kwargs={
                "num_perm": num_perm,
                "ngram_size": num_perm,
                "hashranges": lsh.hashranges,
                "ngram_size": ngram_size,
            },
            input_columns=[column],
            remove_columns=[column],
            num_proc=os.cpu_count(),
            with_indices=True,
            desc=f"Fingerprinting...",
        )
        time_measures["minhash"] = time.time() - time_measures["minhash"]

        # TODO: if one day, the map function supports async mode, we can use it to speed up the following step
        time_measures["clustering"] = time.time()
        uf = UnionFind()
        for record in tqdm(embedded, total=len(embedded), ascii=True):
            key = record["__id__"]
            Hs = record["__signatures__"]
            for H, hashtable in zip(Hs, lsh.hashtables):
                for candidate in hashtable.get(H):
                    uf.union(candidate, key)
            lsh.keys.insert(key, *Hs, buffer=False)
            for H, hashtable in zip(Hs, lsh.hashtables):
                hashtable.insert(H, key, buffer=False)
        time_measures["clustering"] = time.time() - time_measures["clustering"]

        time_measures["filtering"] = time.time()
        ds = ds.map(
            function=lambda _, idx: {"__cluster__": uf.find(idx)},
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Finding clusters...",
        )
        final_data = ds.filter(
            function=lambda record, idx: record["__cluster__"] == idx,
            with_indices=True,
            num_proc=os.cpu_count(),
            desc="Filtering clusters...",
        )
        time_measures["filtering"] = time.time() - time_measures["filtering"]

        time_measures["save"] = time.time()
        final_data = final_data.remove_columns(["__cluster__"])
        final_data.save_to_disk(output)
        time_measures["save"] = time.time() - time_measures["save"]

        FINAL_DATA_SIZE = len(final_data)
        DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
        PAD = 32

        for key, value in time_measures.items():
            logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
        logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
        logger.info(f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})")
        logger.info(f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})")
        logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
        logger.info(f"{'Deduplicated Dataset':<{PAD}}: {output}")
        logger.info("🤗 Happy Deduplicating 🤗")

    typer.run(run)
