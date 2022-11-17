import sys

import pickle
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import time
import datetime

from datasketch import LeanMinHash, MinHashLSH

from dpu_utils.utils.iterators import ThreadedIterator

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util
from text2code_dataset.dataset.postprocessing.near_dedup.lang_size import langs

class ProgressReport():
    def __init__(self, size):
        self.size = size
        self.t0 = time.time()
        
    def start_step(self, index):
        self.index = index
        self.t1 = time.time()
        print(f'{index} / {self.size} {datetime.datetime.utcnow()}')
        
    def finish_step(self):
        t2 = time.time()
        print(f'remained time: {(t2-self.t0)/(self.index+1)*(self.size-self.index-1):.2f}, step time: {(t2-self.t1):.4f}')
        
def get_min_hash_clusters(files):
    index = MinHashLSH(threshold=cfg.jaccard_threshold, num_perm=cfg.NUM_PERM)
    key_to_cluster_map = {}
    pr = ProgressReport(len(files))
    print('--- Builiding index ------------------------------------')
    for i, file in enumerate(files):
        pr.start_step(i)
        for data in util.load_itr(file):
            key = (file.stem, data[0][0], data[0][1])
            key_to_cluster_map[key] = None
            minhash = LeanMinHash.deserialize(data[1])
            index.insert(key, minhash)
        pr.finish_step()
            
    cluster_index = 0
    print('--- Querieng index ------------------------------------')
    pr = ProgressReport(len(files))
    for i, file in enumerate(files):
        pr.start_step(i)
        for data in util.load_itr(file):
            key = (file.stem, data[0][0], data[0][1])
            if key_to_cluster_map[key] is not None:
                continue
            key_to_cluster_map[key] = cluster_index
            minhash = LeanMinHash.deserialize(data[1])
            close_set = index.query(minhash)
            for el in close_set:
                key_to_cluster_map[el] = cluster_index
            cluster_index += 1
        pr.finish_step()
    print('--- Done ------------------------------------')
    return key_to_cluster_map

# def process_minhashes(files, index, duplicate_clusters):
#     cnt = 0
#     t0 = time.time()
#     for i, file in enumerate(files):
#         t1 = time.time()
#         print(f'{i} / {len(files)} {time.time()}')
#         for data in ThreadedIterator(util.load_itr(file), max_queue_size=1000):
#             # data format: ((index in file, path as in cfg.PATH, *(values as in cfg.OTHER_COLUMNS)), serialized min hash)
#             cnt += 1
#             # key format:(bucket file stem, index in file, path as in cfg.PATH)
#             key = (file.stem, data[0][0], data[0][1])
#             if key in index.keys:
#                 print(key)
#                 continue
#             minhash = LeanMinHash.deserialize(data[1])
#             close_set = index.query(minhash)
#             index.insert(key, minhash)
#             if len(close_set) > 0:
#                 for el in close_set:
#                     if el in duplicate_clusters:
#                         duplicate_clusters[el].add(key)
#                         break
#                 else:
#                     duplicate_clusters[close_set[0]].add(key)
#         t2 = time.time()
#         print(f'time from start: {t2-t0}, step time: {t2-t1}')
#     return cnt

def run(lang_index):
    lang_index = int(lang_index)
    lang = langs[lang_index][1]
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

    duplicate_clusters = get_min_hash_clusters(files)

    print('saving')
    with duplicate_clusters_path_tmp.open('wb') as f:
        pickle.dump(duplicate_clusters, f)

    duplicate_clusters_path_tmp.rename(duplicate_clusters_path)

    print('done')


if __name__ == "__main__":
    run(sys.argv[1])