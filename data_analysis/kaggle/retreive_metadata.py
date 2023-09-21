# code for getting metadata based on file id 
import pandas as pd
import numpy as np
import json
from pathlib import Path


code_base_path = Path('/fsx/loubna/kaggle_data/kaggle-code-data/data')
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


def get_metadata(file_id):
    """given the id of a notebook (=the stem of its path) we retrieve metadata from csv tables
    provided by kaggle"""
    output = {}
    kversion = kversions[kversions['Id']==int(file_id)]
    data_source_kernel = kernelversions_datasetsources[kernelversions_datasetsources['KernelVersionId']==int(file_id)]
    source_id = data_source_kernel['SourceDatasetVersionId']
    dataset_v = datasets_versions[datasets_versions['Id']==int(source_id)]
    data_name = dataset_v["Slug"].values[0]
    source_dataset = datasets[datasets['Id']==int(dataset_v["DatasetId"])]
    user = users[users['Id']==int(source_dataset["OwnerUserId"])]
    final_data = f'{user["UserName"].values[0]}/{data_name}'

    output['kaggle_dataset_name'] = data_name
    output['kaggle_dataset_owner'] = user["UserName"].values[0]
    output['kversion'] = kversion.to_dict(orient='records')
    output['kversion_datasetsources'] = data_source_kernel.to_dict(orient='records')
    output['dataset_versions'] = dataset_v.to_dict(orient='records')
    output['datasets'] = source_dataset.to_dict(orient='records')
    output['users'] = user.to_dict(orient='records')

    return output
