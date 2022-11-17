from pathlib import Path
import toolkit_run.dask.apply as apply
import pandas as pd

import text2code_dataset.dataset.postprocessing.near_dedup.cfg as cfg
import text2code_dataset.dataset.postprocessing.near_dedup.util as util


def process(lang, files, dst_path):
    done_file_flag = dst_path / cfg.filename_flag_done
    if done_file_flag.is_file():
        print('done')
        return

    dst_files = dst_path.glob('*.parquet')
    for el in dst_files:
        el.unlink()

    print('lang: ', lang)
    ttl_sz = sum(file.stat().st_size for file in files)
    bucket_size = ttl_sz // cfg.bucket_target_count
    if bucket_size < cfg.bucket_min_size:
        bucket_size = cfg.bucket_min_size
    if bucket_size > cfg.bucket_max_size:
        bucket_size = cfg.bucket_max_size

    cnt_buckets = (ttl_sz // bucket_size) + 1
    bucket_size = ttl_sz // cnt_buckets

    print('bucket_size ', bucket_size)

    index = 0
    sz = 0
    data = []
    for i, file in enumerate(files):
        print('file: ', i, '/', len(files))
        for el, line in util.enum_json_lines(file):
            data.append(el)
            sz += len(line)
            if sz >= bucket_size:
                dst_file = dst_path / f'data_{index:04}.parquet'
                if not dst_file.is_file():
                    dst_file_tmp = dst_path / f'{cfg.filename_prefix_tmp}data_{index:04}.parquet'
                    df = pd.DataFrame(data)
                    df.to_parquet(dst_file_tmp)
                    dst_file_tmp.rename(dst_file)
                print('writen index: ', index)
                index += 1
                data = []
                sz = 0
    
    if len(data) > 0:
        df = pd.DataFrame(data)
        df.to_parquet(dst_path / f'data_{index:04}.parquet')
        print('writen index: ', index)
    
    done_file_flag.touch()



class DRParams(apply.DaskRunParams):
    @property
    def max_cluster_size(self):
        return 11

    def run(self, client):
        src_path = cfg.dst_near_dedup_jsonl_path
        lang_paths = list(src_path.glob('*'))
        #lang_paths = [Path('/repo_workdir/filtered/multi_safe_license_raw/by_lang_min_hash_clusters_data/java')]
        files = {el.stem : list(el.glob('data_*.jsonl')) for el in lang_paths}


        ftrs = []
        for lang, lang_files in files.items():
            dest_path = cfg.dst_near_dedup_parquet_path / lang
            dest_path.mkdir(parents=True, exist_ok=True)
            done_file_flag = dest_path / cfg.filename_flag_done
            if done_file_flag.is_file():
                continue
            ftrs.append(client.submit(
                process,
                lang,
                lang_files,
                dest_path
            ))
        print('waiting')
        client.gather(ftrs)
        print('done')