import pickle
import re
import json
from pathlib import Path

import toolkit_run.dask.apply as apply

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util

from distributed import get_client, secede, rejoin



def get_cluster_elements_to_remove(cluster, jaccard_threshold):
    extrema = util.find_cluster_extrema(cluster, jaccard_threshold)
    cluster_set = set(el[cfg.file_index_cluster_column]+(el[cfg.PATH_COLUMN],) for el in cluster)
    extrema_set = set(el[cfg.file_index_cluster_column]+(el[cfg.PATH_COLUMN],) for el in extrema)
    return cluster_set - extrema_set

def load_clusters(file):
    clusters = {}
    for data in util.load_itr(file):
        for row in data:
            cluster_index = row[cfg.file_index_cluster_column][2]
            row[cfg.CONTENT] = util.get_tokens(row[cfg.CONTENT])
            if cluster_index in clusters:
                clusters[cluster_index].append(row)
            else:
                clusters[cluster_index] = [row]
    return clusters

def process_get_elements_to_remove_for_cluster_bucket(file, jaccard_threshold, dest_path):
    dst_file_id = file.stem
    out_file = dest_path / f'{dst_file_id}.pkl'
    if out_file.is_file():
        return out_file
    
    clusters = load_clusters(file)
    els_to_remove_by_file = {}
    for k, v in clusters.items():
        els_to_remove = get_cluster_elements_to_remove(v, jaccard_threshold)
        for el in els_to_remove:
            file_id = el[0]
            if file_id in els_to_remove_by_file:
                els_to_remove_by_file[file_id].append(el)
            else:
                els_to_remove_by_file[file_id] = [el]

    out_file_tmp = dest_path / f'{cfg.filename_prefix_tmp}{dst_file_id}.pkl'
    with out_file_tmp.open('wb') as f:
        pickle.dump(els_to_remove_by_file, f)
    
    out_file_tmp.rename(out_file)
    return out_file

def filter_data_file(el_to_remove_file, src_data_file, dst_data_file):
    if dst_data_file.is_file():
        return

    to_remove_index = {}
    for data in util.load_itr(el_to_remove_file):
        for el in data:
            index = el[1]
            filename = el[3]
            to_remove_index[index] = filename

    dst_data_file_tmp = dst_data_file.parent / (cfg.filename_prefix_tmp + dst_data_file.name)
    with dst_data_file_tmp.open('wt') as f:
        index = 0
        for data in util.enum_json_lines(src_data_file):
            data, line = data
            if index in to_remove_index:
                assert to_remove_index[index] == data[cfg.PATH_COLUMN]
            else:
                f.write(line)
            index += 1

    dst_data_file_tmp.rename(dst_data_file)

def process_by_cluster_bucket(lang, files, dst_per_data_file, src_data, dst_data):
    dst_per_data_file = dst_per_data_file / lang
    dst_per_data_file.mkdir(parents=True, exist_ok=True)
    dst_per_data_file_done_file_flag = dst_per_data_file / cfg.filename_flag_done
    if not dst_per_data_file_done_file_flag.is_file():
        for file in files:
            for data in util.load_itr(file):
                for file_id, v in data.items():
                    out_file = dst_per_data_file / f'{file_id}.pkl'
                    with out_file.open('ab') as f:
                        pickle.dump(v, f)
        dst_per_data_file_done_file_flag.touch()
    
    dst_data = dst_data / lang
    dst_data.mkdir(parents=True, exist_ok=True)
    files = list(dst_per_data_file.glob('data_*.pkl'))
    ftrs = []
    client = get_client()
    for file in files:
        src_data_file = src_data / lang / f'{file.stem}.jsonl'
        dst_data_file = dst_data / f'{file.stem}.jsonl'
        ftrs.append(client.submit(filter_data_file, file, src_data_file, dst_data_file))

    secede()
    client.gather(ftrs)
    rejoin()

class DRParams(apply.DaskRunParams):
    @property
    def max_cluster_size(self):
        return 100

    def run(self, client):
        src_path = cfg.dst_min_hash_clusters_data_path
        lang_paths = list(src_path.glob('*'))
        #lang_paths = [Path('/repo_workdir/filtered/multi_safe_license_raw/by_lang_min_hash_clusters_data/java')]
        files = {el.stem : list(el.glob('*.pkl')) for el in lang_paths}

        # compute pairwise distance and return elements to remove per cluster bucket
        res_f_by_lang = {}
        for lang, v in files.items():
            dest_path = cfg.dst_el_to_remove_by_cluster_bucket_path / lang
            dest_path.mkdir(parents=True, exist_ok=True)
            ftrs = []
            for file in v:
                ftrs.append(client.submit(
                    process_get_elements_to_remove_for_cluster_bucket,
                    file,
                    cfg.jaccard_threshold,
                    dest_path
                ))
            res_f_by_lang[lang] = ftrs

        # rearrange elements to remove from per cluster bucket to per data file

        res_f_by_data_file = []
        for lang, ftrs in res_f_by_lang.items():
            res_f_by_data_file.append(client.submit(
                process_by_cluster_bucket,
                lang,
                ftrs,
                cfg.dst_el_to_remove_by_data_file_path,
                cfg.src_jsonl_path,
                cfg.dst_near_dedup_jsonl_path
            ))

        client.gather(res_f_by_data_file)
