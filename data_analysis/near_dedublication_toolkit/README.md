## Near deduplication with min hash lsh index
Code taken from https://github.com/bigcode-project/bigcode-analysis/tree/main/data_analysis/near-deduplication and changed to be run on distributed Dask cluster and independent nodes with max RAM usage as for now under 600GB but with potential to be reduced even more if to distribute minhash lsh index itself in step 2 and to divide minhash cluster buckets additionally according to cluster element count, not just cluster number in step 3.

Right now the code is in 5 steps which are to be run manually each. However, pipeline can be fully automated either by reducing memory requirements as described above or if to implement dask worker nodes with different resources on toolkit (dask itself allows it)

- `cfg.py` contains configuration
- `util.py` contains most of the code taken from the above repo
- `[step_number]_*.py` contains step code
- `[step_number]_*.child.yaml` contains resources of a worker node if a step is a Dask cluster step

 ### 1 step
Compute min hashes for all data and stores them to files. Runs as Dask cluster on toolkit. 
```
make text2code_dataset/dataset/postprocessing/near_dedup/1_get_min_hashes.run-slim-dask MORE_JOB_ARGS="--data snow.code_llm.data:/data"
```

### 2 step
Computes minhash clusters of potential near duplicates files. Runs as separate jobs on toolkit for each language with different RAM size depending on language data amount. With incremental index/query build takes up to 24 hours for html language.
TODO: implement either one off index build and retrieve similarly to how Chenghao Mou did https://github.com/bigcode-project/bigcode-analysis/pull/11 for faster run time. Or implement distributed index to be run as Dask cluster
```
python text2code_dataset/dataset/postprocessing/near_dedup/2_launchlocal_get_min_hash_clusters.py
```

### 3 step
Adds data to the clusters and saves them in buckets. Runs as separate toolkit jobs. TODO: split bigger buckets so they are equal in size and so that next step can run without interruption
```
python text2code_dataset/dataset/postprocessing/near_dedup/3_launchlocal_group_duplicate_cluster_data.py
```
 
### 4 step
Computes pairwise Jaccard similarity within clusters and identifies near duplicates to remove. Rearranges row to remove form per cluster to per data files as in source file split and removes them from source files. Runs as dask task graph on dask cluster. Currently need manual re-run for one cluster bucket for html as it is too big to fit to worker memory (see notes for previous step)
```
make text2code_dataset/dataset/postprocessing/near_dedup/4_remove_near_duplicates_by_clusters.run-slim-dask  MORE_JOB_ARGS="--data snow.code_llm.data:/data"
```

### 5 step
Redistribute buckets and write them as parquet. Runs on Dask cluster
```
make text2code_dataset/dataset/postprocessing/near_dedup/5_rearrange.run-slim-dask  MORE_JOB_ARGS="--data snow.code_llm.data:/data"

```