import re
import json
import pickle

from datasketch import MinHash, LeanMinHash
from typing import Dict, List, Optional, Set, Tuple, Type

from dpu_utils.utils.iterators import ThreadedIterator
import multiprocessing as mp

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg

NON_ALPHA = re.compile("[^A-Za-z_0-9]")

def enum_json_lines(file):
    with open(file, "rt") as f:
        for line in f:
            try:
                jline = json.loads(line)
                yield jline, line
            except Exception:
                continue

def load_itr(file):
    with file.open('rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def get_tokens(code):
    """Tokenize a code snippet."""
    return set([t for t in NON_ALPHA.split(code) if len(t.strip()) > 0])

def jaccard_similarity(tokens1: set, tokens2: set) -> float:
    """Compute the Jaccard similarity of two code snippets."""
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def find_cluster_extrema(cluster, jaccard_threshold):
    extremes = []
    for element1 in cluster:
        code1 = element1[cfg.CONTENT]
        for element2 in extremes:
            code2 = element2[cfg.CONTENT]
            if jaccard_similarity(code1, code2) >= jaccard_threshold:
                element2["copies"] += 1
                break
        else:
            element1["copies"] = 1
            extremes.append(element1)
    return extremes

def _compute_min_hash(element):
    index, data = element
    data = data[0]

    min_hash = get_min_hash([t for t in NON_ALPHA.split(data[cfg.CONTENT]) if len(t.strip()) > 0])
    if min_hash is not None:
        return (index, data[cfg.PATH_COLUMN]), min_hash

def get_min_hash(tokens: List[str]):
    """Compute the MinHash of a code snippet."""
    if len(tokens) < cfg.MIN_NUM_TOKENS:
        return None
    min_hash = MinHash(num_perm=cfg.NUM_PERM)
    for token in set(tokens):
        min_hash.update(token.encode())
    min_hash = LeanMinHash(min_hash)
    buf = bytearray(min_hash.bytesize())
    min_hash.serialize(buf)
    return buf

def get_tokens(code: str) -> Set[str]:
    """Tokenize a code snippet."""
    return set([t for t in NON_ALPHA.split(code) if len(t.strip()) > 0])

def minhash_iter(dataset_iterator):
    with mp.Pool() as pool:
        for data in pool.imap_unordered(
            _compute_min_hash,
            ThreadedIterator(dataset_iterator, max_queue_size=10000),
            chunksize=100,
        ):
            if data is not None:
                yield data