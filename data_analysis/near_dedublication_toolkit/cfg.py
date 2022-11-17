from pathlib import Path

MIN_NUM_TOKENS = 10
NUM_PERM = 256
jaccard_threshold = 0.85

src_jsonl_path = Path('/data/filtering/test11d_stack_v1_1/data/')

tmp_root = Path('/repo_workdir/filtered/the_stack_v1_1')

dst_min_hashes_path = tmp_root / 'by_lang_min_hashes/'
dst_min_hash_clusters_path = tmp_root / 'by_lang_min_hash_clusters'
filename_min_hash_clusters = 'duplicate_clusters.pkl'
dst_min_hash_clusters_data_path = tmp_root /  'by_lang_min_hash_clusters_data'
dst_el_to_remove_by_cluster_bucket_path = tmp_root / 'by_lang_els_to_remove_by_cluster_buckets'
dst_el_to_remove_by_data_file_path = tmp_root /  'by_lang_els_to_remove_by_data_file/'

dst_near_dedup_jsonl_path = Path('/data/hf_repos/the_stack_v1_1_near_dedup/data/')
dst_near_dedup_parquet_path = Path('/data/hf_repos/the_stack_v1_1_near_dedup_parquet/data/')

CONTENT = "content"
PATH_COLUMN = "max_stars_repo_path"
OTHER_COLUMNS = [
    'max_stars_repo_name', 'max_stars_repo_head_hexsha',
    'max_stars_repo_licenses', 'max_stars_count',
    'max_stars_repo_stars_event_min_datetime',
    'max_stars_repo_stars_event_max_datetime', 'max_issues_repo_path',
    'max_issues_repo_name', 'max_issues_repo_head_hexsha',
    'max_issues_repo_licenses', 'max_issues_count',
    'max_issues_repo_issues_event_min_datetime',
    'max_issues_repo_issues_event_max_datetime', 'max_forks_repo_path',
    'max_forks_repo_name', 'max_forks_repo_head_hexsha',
    'max_forks_repo_licenses', 'max_forks_count',
    'max_forks_repo_forks_event_min_datetime',
    'max_forks_repo_forks_event_max_datetime'
]
file_index_cluster_column = '__file_index_cluster__'

num_cluster_buckets = 256

bucket_target_count = 100
bucket_min_size = 100 * 1024 * 1024
bucket_max_size = 3024 * 1024 * 1024


filename_flag_done = '__done__'
filename_prefix_tmp = '__tmp__'