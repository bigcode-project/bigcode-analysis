import pickle
from pathlib import Path
from distributed import as_completed

from dpu_utils.utils.iterators import ThreadedIterator

import toolkit_run.dask.apply as apply

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util

import os 



def process(index, file, dst_path):
    parent = file.parent.name
    file_stem = file.stem
    dst_path = dst_path / parent
    dst_path.mkdir(parents=True, exist_ok=True)

    # enountered tmp file beeing absent at the end, so to distinguis double task run form something else
    #   add random to tmp prefix and then try to rename 
    dst_file_tmp = dst_path / f'{cfg.filename_prefix_tmp}__{os.urandom(8).hex()}__{file_stem}.pkl'
    dst_file = dst_path / f'{file_stem}.pkl'

    if dst_file.is_file():
        return

    with dst_file_tmp.open('wb') as f:
        for data in ThreadedIterator(
            util.minhash_iter(enumerate(util.enum_json_lines(file))), max_queue_size=100
        ):
            # data format: ((index in file, path as in cfg.PATH, *(values as in cfg.OTHER_COLUMNS)), serialized min hash)
            pickle.dump(data, f)

    try:
        dst_file_tmp.rename(dst_file)
    except Exception:
        if dst_file.is_file():
            return file
        raise

def get_not_processed_files():
    dst_path = cfg.dst_min_hashes_path
    print('cleaning')
    tmp_files = list(dst_path.glob(f'*/{cfg.filename_prefix_tmp}*.pkl'))
    for tmp_file in tmp_files:
        tmp_file.unlink()

    print('get src files')
    files = list(cfg.src_jsonl_path.glob('*/*.jsonl'))
    files = [f'{file.parent.name}/{file.stem}' for file in files]
    files  = set(files)
    print('files count', len(files))
    
    print('get processed files')
    processed_files = list(cfg.dst_min_hashes_path.glob('*/*.pkl'))
    processed_files = [f'{file.parent.name}/{file.stem}' for file in processed_files]
    processed_files  = set(processed_files)
    print('processed files count', len(processed_files))
    
    # Must be no files in processed which are not in src
    assert len(processed_files.difference(files)) == 0
    
    not_processed = files.difference(processed_files)
    
    not_processed = [cfg.src_jsonl_path / f'{file}.jsonl' for file in not_processed]
    print('not processed files count', len(not_processed))
    return not_processed


class DRParams(apply.DaskRunParams):
    @property
    def max_cluster_size(self):
        return 1

    def run(self, client):
        cfg.dst_min_hashes_path.mkdir(parents=True, exist_ok=True)
        files = get_not_processed_files()
        
        res_f = []
        print('submitting')
        for index, file in enumerate(files):
            res_f.append(client.submit(process, index, file, cfg.dst_min_hashes_path))
        
        print('waiting')
        complete_cnt = 0
        for f, r in as_completed(res_f, with_results=True, raise_errors=True):
            complete_cnt += 1
            if complete_cnt % 100 == 0:
                print(complete_cnt, '/', len(res_f))
            if r is not None:
                print(r)
            error = f.exception()
            if error is not None:
                print(error)
                return