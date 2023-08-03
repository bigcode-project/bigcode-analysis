from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset


source = "https://huggingface.co/datasets/bigcode-data/pr_code_reviews/resolve/main/groupby_pull_request_id"

def process_bucket(bucket_id="00"):
    """Build a dataset with fours cols: pull_request_info, head_repo_info, base_repo_info and events (list of dicts/events) with only main info"""
    pr_00 = load_dataset(
        "parquet",
        data_files=[f"{source}/{bucket_id}.parquet"],
        split="train",
    )
    df_00 = pr_00.to_pandas()
    print(f"number of PRs in bucket {bucket_id}: {len(df_00['pull_request.id'].unique())}")

    # group by pull_request.id
    grouped_data = df_00.groupby("pull_request.id")

    # we dropped some columns that didn't seem useful
    pull_request_info_cols = ['repo.name', 'public', 'pull_request.id', 'pull_request.number', 'pull_request.title', 'pull_request.body', 'pull_request.state', 'pull_request.user.login', 'pull_request.user.id', 'pull_request.created_at', 'pull_request.closed_at', 'pull_request.merged_at']
    head_info_cols =  ['pull_request.head.label', 'pull_request.head.ref', 'pull_request.head.user.login', 'pull_request.head.user.type', 'pull_request.head.repo.owner.login', 'pull_request.head.repo.owner.type', 'pull_request.head.repo.license.name', 'pull_request.head.sha']
    base_info_cols = ['pull_request.base.label', 'pull_request.base.ref', 'pull_request.base.sha', 'pull_request.base.user.login', 'pull_request.base.user.type', 'pull_request.base.repo.owner.login', 'pull_request.base.repo.owner.type', 'pull_request.base.repo.license.name', 'pull_request.base.repo.default_branch','pull_request.base.repo.description', 'pull_request.base.repo.language', 'pull_request.base.repo.watchers_count', 'pull_request.base.repo.open_issues_count', 'pull_request.base.repo.forks_count']
    pull_request_cols = pull_request_info_cols + head_info_cols + base_info_cols
    # drop "repo.name", "repo.id", "public" so they are not duplicated and keep relevant columns that might change
    event_cols =[col for col in df_00.columns if (not col.startswith("pull_request.")) and col not in ["repo.name", "repo.id", "public"]] + ['pull_request.head.label', 'pull_request.head.ref', 'pull_request.head.sha', 'pull_request.title']

    pr_dict = []
    for name, group in tqdm(grouped_data):
        group_dict = {}
        # sort events by date created_at
        group = group.sort_values(by=["created_at"])
        events = group.to_dict(orient="records")
        # we get info from the first event
        group_dict["pull_request_info"] = group[pull_request_info_cols].to_dict(orient="records")[0]
        group_dict["head_repo_info"] = group[head_info_cols].to_dict(orient="records")[0]
        group_dict["base_repo_info"]  = group[base_info_cols].to_dict(orient="records")[0]
        group_dict["events"] = group[event_cols].to_dict(orient="records")
        pr_dict.append(group_dict)
    return pr_dict

# process 3 random buckets
pr_dict_00 = process_bucket("031")
pr_dict_01 = process_bucket("093")
pr_dict_02 = process_bucket("02B")
pr_dict_03 = process_bucket("005")

buckets = pr_dict_00 + pr_dict_01 + pr_dict_02 + pr_dict_03
hf_dataset = Dataset.from_pandas(pd.DataFrame(data=buckets))
print(hf_dataset)
hf_dataset.push_to_hub("pr_code_reviews_sample")
