#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-04 12:09:38
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import glob
import os
import time
from typing import List, Tuple

import dask.dataframe as dd
import pyarrow as pa
import regex
import typer
from dask.distributed import Client
from datasketch import MinHash, MinHashLSH
from loguru import logger
from nltk import ngrams
from tqdm import tqdm

# TODO There are always some pending tasks at the end. I cannot find a way to solve it.

NON_ALPHA = regex.compile("[^A-Za-z_0-9]")
INDEX = "__id__"
CLUSTER = "__cluster__"


def compuate_minhash(
    record,
    hashranges: List[Tuple[int]],
    ngram_size: int,
    num_perm: int,
    seed: int,
) -> List:
    idx, content = record
    m = MinHash(num_perm=num_perm, seed=seed)
    tokens = NON_ALPHA.split(content)
    m.update_batch([token.encode("utf-8") for token in {" ".join(t) for t in ngrams(tokens, ngram_size)}])
    Hs = [bytes(m.hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return idx, Hs


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.parent[self.find(y)] = min(self.find(x), self.find(y))


if __name__ == "__main__":

    def run(
        input_files: str = typer.Argument(..., help="Input file pattern"),
        threshold: float = typer.Option(0.7, help="LSH Threshold"),
        num_perm: int = typer.Option(256, help="Number of permutations"),
        ngram_size: int = typer.Option(5, help="Ngram size"),
        seed: int = typer.Option(42, help="Seed"),
        column: str = typer.Option("content", help="Code column"),
        num_partitions: int = typer.Option(10, help="Number of partitions"),
        output: str = typer.Option("output", help="Output file"),
    ):
        start_time = time.time()
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        uf = UnionFind()
        client = Client(n_workers=os.cpu_count(), threads_per_worker=2, memory_limit="60GB", asynchronous=False)
        df = dd.read_parquet(glob.glob(input_files)).repartition(npartitions=os.cpu_count())
        df[INDEX] = df.assign(partition_count=1).partition_count.cumsum()
        df = df.persist()
        results = df[[INDEX, column]].apply(
            compuate_minhash,
            hashranges=lsh.hashranges,
            ngram_size=ngram_size,
            num_perm=num_perm,
            seed=seed,
            axis=1,
            meta=(None, "object"),
        )
        minhashes = results.compute(allow_other_workers=True, schedule="processes", sync=True)
        for key, Hs in tqdm(minhashes, total=len(minhashes), ascii=True):
            for H, hashtable in zip(Hs, lsh.hashtables):
                for candidate in hashtable.get(H):
                    uf.union(candidate, key)
            lsh.keys.insert(key, *Hs, buffer=False)
            for H, hashtable in zip(Hs, lsh.hashtables):
                hashtable.insert(H, key, buffer=False)
        df[CLUSTER] = df[INDEX].apply(uf.find, meta=(CLUSTER, "int64"))
        df = df.drop(columns=[INDEX])
        df = df.drop_duplicates(subset=[CLUSTER], keep="first", ignore_index=True)
        df = df.drop(columns=[CLUSTER])
        df = df.repartition(npartitions=num_partitions).persist()
        logger.info(f"Dataset size after deduplication: {len(df):,}")
        df.to_parquet(
            output,
            schema={
                "max_stars_repo_licenses": pa.list_(pa.string()),
                "max_issues_repo_licenses": pa.list_(pa.string()),
                "max_forks_repo_licenses": pa.list_(pa.string()),
            },
            compute=True,
            compute_kwargs={"schedule": "processes", "allow_other_workers": True, "sync": True},
        )
        logger.info(f"Total time: {time.time() - start_time:.2f}s")

    typer.run(run)
