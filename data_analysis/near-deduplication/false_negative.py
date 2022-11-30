#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-28 19:35:07
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import gc
import multiprocessing
import os
import random
import re

import numpy as np

# import datasets
# from mpire import WorkerPool
from datasets import load_from_disk

# from datasets.utils import disable_progress_bar
from rich.console import Console
from tqdm import tqdm

multiprocessing.set_start_method("fork", force=True)
# disable_progress_bar()
# datasets.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()

tokens = None


def set_jaccard_similarity(a, b):
    return len(a.intersection(b)) / len(a.union(b))


if __name__ == "__main__":

    import typer

    def run(
        dataset: str = typer.Option(None, help="Path to results"),
        sample_size: int = typer.Option(2_000, help="Number of samples"),
    ):
        global tokens
        ds = load_from_disk(dataset)
        N = min(sample_size, len(ds))
        samples = ds.train_test_split(test_size=N)["test"]
        samples = samples.map(
            lambda r, idx: {"tokens": {t for t in NON_ALPHA.split(r["content"]) if t}, "idx": idx},
            num_proc=os.cpu_count(),
            desc="Tokenizing",
            with_indices=True,
        )
        tokens = [set(t) for t in samples["tokens"]]

        def calc(record):
            results = []
            for j in range(record["idx"] + 1, len(tokens)):
                results.append(set_jaccard_similarity(set(record["tokens"]), tokens[j]))
            return {"idx": record["idx"], "results": results}

        gc.disable()
        gc.freeze()
        samples = samples.map(
            calc,
            num_proc=os.cpu_count(),
            desc="Calculating",
        )
        gc.enable()
        gc.collect()

        matrix = np.zeros((N, N))
        for r in tqdm(samples, desc="Building matrix"):
            for j, v in enumerate(r["results"], start=r["idx"] + 1):
                matrix[r["idx"], j] = matrix[j, r["idx"]] = v

        scores = matrix.max(axis=1)
        for threshold in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
            TP = np.sum(scores >= threshold)
            console.print(f"Threshold: {threshold} | TP: {TP} ({TP/N:.2%})")

    typer.run(run)
