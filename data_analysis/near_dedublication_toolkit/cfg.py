from pathlib import Path

MIN_NUM_TOKENS = 10
NUM_PERM = 256
jaccard_threshold = 0.85

src_jsonl_path = Path('/data/hf_repos/multi_safe_license_raw/data/')
dst_min_hashes_path = Path(f'/repo_workdir/filtered/multi_safe_license_raw/by_lang_min_hashes/')
dst_min_hash_clusters_path = Path('/repo_workdir/filtered/multi_safe_license_raw/by_lang_min_hash_clusters')
filename_min_hash_clusters = 'duplicate_clusters.pkl'
dst_min_hash_clusters_data_path = Path('/repo_workdir/filtered/multi_safe_license_raw/by_lang_min_hash_clusters_data')
dst_el_to_remove_by_cluster_bucket_path = Path(f'/repo_workdir/filtered/multi_safe_license_raw/by_lang_els_to_remove_by_cluster_buckets')
dst_el_to_remove_by_data_file_path = Path('/repo_workdir/filtered/multi_safe_license_raw/by_lang_els_to_remove_by_data_file/')
dst_near_dedup_jsonl_path = Path('/data/hf_repos/multi_safe_license_raw_near_dedup/data/')
dst_near_dedup_parquet_path = Path('/data/hf_repos/multi_safe_license_raw_near_dedup_parquet/data/')

CONTENT = "content"
PATH_COLUMN = "path"
file_index_cluster_column = '__file_index_cluster__'

num_cluster_buckets = 256

bucket_target_count = 100
bucket_min_size = 100 * 1024 * 1024
bucket_max_size = 3024 * 1024 * 1024


filename_flag_done = '__done__'
filename_prefix_tmp = '__tmp__'