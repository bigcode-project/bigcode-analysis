#!/usr/bin/env python
# coding: utf-8
# this script replicate `minhash_deduplication.py` for the clustering part.
import multiprocessing as mp
import os

import networkit as nk
from dpu_utils.utils.iterators import ThreadedIterator
from tqdm import tqdm

from minhash_deduplication_alt import ngrams
from minhash_deduplication import CONTENT, NON_ALPHA, DuplicationIndex, get_min_hash, minhash_iter
NGRAM_SIZE = 1

def _compute_min_hash(element):
    index, data = element
    min_hash = get_min_hash([" ".join(t) for t in ngrams(NON_ALPHA.split(data[CONTENT]), NGRAM_SIZE)])
    if min_hash is not None:
        return index, min_hash


def minhash_iter(dataset_iterator):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
            _compute_min_hash,
            ThreadedIterator(dataset_iterator, max_queue_size=10000),
            chunksize=100,
        ):
            if data is not None:
                yield data


def make_duplicate_clusters(dataset_iterator, jaccard_threshold: float):
    """Find duplicate clusters in the dataset in two steps:
    1. Compute MinHash for each code snippet. MinHash is a tool for fast jaccard similarity estimation.
    This step is computed using an asynchronous multiprocessing pool, minhash_iter
    2. Find duplicate clusters. The computed MinHash is added sequentially to the DuplicationIndex.
    This step cannot be parallelized. So using asynchronous thread in the previous step helps to speed up the process.
    """
    di = DuplicationIndex(duplication_jaccard_threshold=jaccard_threshold)

    for filename, min_hash in tqdm(ThreadedIterator(minhash_iter(enumerate(dataset_iterator)), max_queue_size=100)):
        di.add(filename, min_hash)

    duplicate_clusters = []
    for base, duplicates in di._duplicate_clusters.items():
        cluster = [base] + list(duplicates)
        cluster = [el for el in cluster]
        duplicate_clusters.append(cluster)
    return duplicate_clusters


def _directory_find(goal, root="."):
    for path, dirs, _ in os.walk(root):
        if goal in dirs:
            return os.path.join(path, goal)
    raise FileNotFoundError(f"Could not find {goal} in {root}")


if __name__ == "__main__":

    from datasets import load_from_disk

    import typer

    def run(
        dataset_root: str = typer.Option(..., help="Dataset name"),
        threshold: float = typer.Option(0.85, help="Jaccard similarity threshold"),
        ngram_size: int = typer.Option(5, help="The ngram size to use for MinHash"),
    ):
        global NGRAM_SIZE
        NGRAM_SIZE = ngram_size
        dataset = _directory_find("indexed", dataset_root.rstrip("/") + "/" + str(int(threshold * 100)))
        ds = load_from_disk(dataset)
        clusters = make_duplicate_clusters(ds, threshold)
        g = nk.graph.Graph()
        for cluster in clusters:
            for x, y in zip(cluster[:-1], cluster[1:]):
                g.addEdge(x, y, addMissing=True)

        output_graph = dataset.rstrip("/").replace("indexed", "alternative.networkit")
        if os.path.exists(output_graph):
            os.remove(output_graph)
        nk.writeGraph(g, str(output_graph), nk.Format.NetworkitBinary)
        print(f"Saved graph to {output_graph}")

    typer.run(run)