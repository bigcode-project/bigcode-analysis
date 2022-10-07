import sys

import pickle
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import time

from datasketch import LeanMinHash, MinHashLSH

from dpu_utils.utils.iterators import ThreadedIterator

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util

def process_minhashes(files, index, duplicate_clusters):
    cnt = 0
    t0 = time.time()
    for i, file in enumerate(files):
        t1 = time.time()
        print(f'{i} / {len(files)} {time.time()}')
        for data in ThreadedIterator(util.load_itr(file), max_queue_size=1000):
            cnt += 1
            key = (file.stem, data[0][0], data[0][1])
            if key in index.keys:
                print(key)
                continue
            minhash = LeanMinHash.deserialize(data[1])
            close_set = index.query(minhash)
            index.insert(key, minhash)
            if len(close_set) > 0:
                for el in close_set:
                    if el in duplicate_clusters:
                        duplicate_clusters[el].add(key)
                        break
                else:
                    duplicate_clusters[close_set[0]].add(key)
        t2 = time.time()
        print(f'time from start: {t2-t0}, step time: {t2-t1}')
    return cnt

def run(lang):
    print('language: ', lang)

    dst_path = cfg.dst_min_hash_clusters_path / lang
    dst_path.mkdir(parents=True, exist_ok=True)
    duplicate_clusters_path = dst_path / cfg.filename_min_hash_clusters
    duplicate_clusters_path_tmp = dst_path / f'{cfg.filename_prefix_tmp}{cfg.filename_min_hash_clusters}'
    print('destination: ', duplicate_clusters_path)
    if duplicate_clusters_path.is_file():
        print('file exist, done')
        return

    files = list((cfg.dst_min_hashes_path / lang).glob('*.pkl'))
    print('src files[0]: ', files[0])
    index = MinHashLSH(threshold=cfg.jaccard_threshold, num_perm=cfg.NUM_PERM)
    duplicate_clusters = defaultdict(set)

    print('processing')
    process_minhashes(files, index, duplicate_clusters)

    print('saving')
    with duplicate_clusters_path_tmp.open('wb') as f:
        pickle.dump(duplicate_clusters, f)

    duplicate_clusters_path_tmp.rename(duplicate_clusters_path)

    print('done')


if __name__ == "__main__":
    run(sys.argv[1])