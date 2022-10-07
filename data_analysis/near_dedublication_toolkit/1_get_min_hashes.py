import pickle
from pathlib import Path
from distributed import as_completed

from dpu_utils.utils.iterators import ThreadedIterator

import toolkit_run.dask.apply as apply

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util


def process(index, file, dst_path):
    parent = file.parent.leaf
    file_stem = file.stem
    dst_path = dst_path / parent
    dst_path.mkdir(parents=True, exist_ok=True)

    dst_file_tmp = dst_path / f'{cfg.filename_prefix_tmp}{file_stem}.pkl'
    dst_file = dst_path / f'{file_stem}.pkl'

    if dst_file.is_file():
        return

    with dst_file_tmp.open('wb') as f:
        for data in ThreadedIterator(
            util.minhash_iter(enumerate(util.enum_json_lines(file))), max_queue_size=100
        ):
            pickle.dump(data, f)

    dst_file_tmp.rename(dst_file)


class DRParams(apply.DaskRunParams):
    @property
    def max_cluster_size(self):
        return 100

    def run(self, client):
        dst_path = cfg.dst_min_hashes_path
        dst_path.mkdir(parents=True, exist_ok=True)
        print('cleaning')
        tmp_files = list(dst_path.glob(f'*/{cfg.filename_prefix_tmp}*.pkl'))
        for tmp_file in tmp_files:
            tmp_file.unlink()

        print('globbing')
        files = list(cfg.src_jsonl_path.glob('*/*.jsonl'))
        print('files', len(files))
        
        res_f = []
        print('submitting')
        for index, file in enumerate(files):
            res_f.append(client.submit(util.process, index, file, dst_path))
        
        print('waiting')
        for f, r in as_completed(res_f, with_results=True, raise_errors=True):
            pass