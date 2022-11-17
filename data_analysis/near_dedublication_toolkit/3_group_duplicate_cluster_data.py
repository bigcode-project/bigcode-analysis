from pathlib import Path
import pickle
import json
import re
import shutil
import sys

from collections import defaultdict

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util
from text2code_dataset.dataset.postprocessing.near_dedup.lang_size import langs

from dpu_utils.utils.iterators import ThreadedIterator

                
def group_data_to_clusters(files, map_el_to_cluster, dest_path, num_cluster_buckets):
    dest_path = Path(dest_path)

    dest_path.mkdir(parents=True, exist_ok=True)
    for file_index, file in enumerate(files):
        print(file_index, '/', len(files))
        file_id = file.stem
        
        map_file_el_to_cluster = map_el_to_cluster[file_id]
        index = 0
        clusters = {}
        
        for data in ThreadedIterator(util.enum_json_lines(file), max_queue_size=1000):
            data, line = data
            if not index in map_file_el_to_cluster:
                # unique files without minhash duplicates are not included into the cluster map
                index += 1
                continue
            assert map_file_el_to_cluster[index][1] == data[cfg.PATH_COLUMN]
            cluster_index = map_file_el_to_cluster[index][0]
            data[cfg.file_index_cluster_column] = (file_id, index, cluster_index)
            cluster_bucket = cluster_index % num_cluster_buckets
            if cluster_bucket in clusters:
                clusters[cluster_bucket].append(data)
            else:
                clusters[cluster_bucket] = [data]
            index += 1
        for k, v in clusters.items():
            out_file = dest_path / f'{k}.pkl'
            with out_file.open('ab') as f:
                pickle.dump(v, f)

def get_clusters_no_singles(min_hash_clusters):
    clusters = defaultdict(list)
    for k, v in min_hash_clusters.items():
        clusters[v].append(k)
    return {k: v for k, v in clusters.items() if len(v) > 1}

def get_el_to_cluster_map(clusters):
    map_el_to_cluster = defaultdict(lambda:defaultdict(tuple))
    for k, v in clusters.items():
        for el in v:
            map_el_to_cluster[el[0]][el[1]] = (k, el[2])
    return map_el_to_cluster

def run(lang_index):
    lang_index = int(lang_index)
    lang = langs[lang_index][1]
    print(lang)

    dest_path = cfg.dst_min_hash_clusters_data_path / lang
    done_flag_file  = dest_path / cfg.filename_flag_done
    if done_flag_file.is_file():
        print('operation completed, done')
        return

    print('cleaning destination', dest_path)
    shutil.rmtree(dest_path, ignore_errors=True)

    duplicate_clusters_path = cfg.dst_min_hash_clusters_path / lang / cfg.filename_min_hash_clusters
    print('src: ', duplicate_clusters_path)

    print('loading duplicates')
    with duplicate_clusters_path.open('rb') as f:
        duplicate_clusters = pickle.load(f)

    print('number of clusters', len(duplicate_clusters))

    print('get_clusters_no_singles')
    clusters = get_clusters_no_singles(duplicate_clusters)
    print('get_el_to_cluster_map')
    map_el_to_cluster = get_el_to_cluster_map(clusters)

    files = list(Path(cfg.src_jsonl_path / lang).glob('*.jsonl'))
    print('number of data files: ', len(files))

    print('group_data_to_clusters')
    group_data_to_clusters(
        files,
        map_el_to_cluster,
        dest_path,
        cfg.num_cluster_buckets
    )
    done_flag_file.touch()
    print('done')

if __name__ == "__main__":
    run(sys.argv[1])

    