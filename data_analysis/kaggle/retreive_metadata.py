# code for getting metadata based on file id 
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datasets import load_dataset, Dataset

ds = load_dataset("bigcode/kaggle-notebooks-data", use_auth_token=True, split="train")
print("dataset loaded")

kv_csv = '/fsx/loubna/kaggle_data/metadata_kaggle/KernelVersions.csv'
kernelversions_datasetsources_csv = '/fsx/loubna/kaggle_data/metadata_kaggle/KernelVersionDatasetSources.csv'
datasets_versions_csv = '/fsx/loubna/kaggle_data/metadata_kaggle/DatasetVersions.csv'
datasets_csv = '/fsx/loubna/kaggle_data/metadata_kaggle/Datasets.csv'
users_csv = '/fsx/loubna/kaggle_data/metadata_kaggle/Users.csv'

kversions = pd.read_csv(kv_csv)
datasets_versions = pd.read_csv(datasets_versions_csv)
datasets = pd.read_csv(datasets_csv)
kernelversions_datasetsources = pd.read_csv(kernelversions_datasetsources_csv)
users = pd.read_csv(users_csv)
print("metadata loaded")

def safe_get(dataframe, condition, column=None):
    """Utility function to safely get value from DataFrame."""
    result = dataframe[condition]
    if result.empty:
        return None
    if column:
        return result[column].values[0]
    return result

def get_metadata(file_id):
    """given the id of a notebook (=the stem of its path) we retrieve metadata from csv tables
    provided by kaggle"""

    file_id_int = int(file_id)
    kversion = safe_get(kversions, kversions['Id'] == file_id_int)
    data_source_kernel = safe_get(kernelversions_datasetsources, kernelversions_datasetsources['KernelVersionId'] == file_id_int)
    
    source_id = None if data_source_kernel is None else data_source_kernel['SourceDatasetVersionId'].values[0]
    dataset_v = safe_get(datasets_versions, datasets_versions['Id'] == source_id)
    
    data_name = dataset_v["Slug"].values[0] if dataset_v is not None else None
    dataset_id = dataset_v["DatasetId"].values[0] if dataset_v is not None else None
    
    source_dataset = safe_get(datasets, datasets['Id'] == dataset_id)
    owner_user_id = None if source_dataset is None else source_dataset["OwnerUserId"].values[0]
    
    user = safe_get(users, users['Id'] == owner_user_id)
    user_name = None if user is None else user["UserName"].values[0]

    return {
        'kaggle_dataset_name': data_name,
        'kaggle_dataset_owner': user_name,
        'kversion': json.dumps(kversion.to_dict(orient='records')) if kversion is not None else None,
        'kversion_datasetsources': json.dumps(data_source_kernel.to_dict(orient='records')) if data_source_kernel is not None else None,
        'dataset_versions': json.dumps(dataset_v.to_dict(orient='records')) if dataset_v is not None else None,
        'datasets': json.dumps(source_dataset.to_dict(orient='records')) if source_dataset is not None else None,
        'users': json.dumps(user.to_dict(orient='records')) if user is not None else None
    }


def retrive_metadata(row):
    output = get_metadata(row['file_id'])
    return output

# issue when using map with multipprocessing new values are None
new_ds = ds.map(retrive_metadata)
new_ds.push_to_hub("kaggle-notebooks-data-metadata-20k")
