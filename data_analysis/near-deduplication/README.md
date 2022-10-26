# Near deduplication

Code for running near-deduplication with MinHash and LSH indexing

### Setup

````
pip install -r requirements.txt
````

Login to be able to be able to push the dataset to the hub after deduplication and clone your huggingface-hub repositories:

````
huggingface-cli login
````

And make sure you have git-lfs installed.

If you use datasets with different column names from the BigCode ones, you might need to change `PATH_COLUMN` and `CONTENT` variables in `minhash_deduplication.py`.

### Usage

To run near deduplication use the following command and adapt the arguments for your case:

````
python near_deduplicate.py \
    --dataset_name bigcode-data/python_any_license_v2 \
    --org bigcode-data \
    --repo_name python_any_license_v2_near_dedup \
    --out_path ./data/any_license-near-dedup \
    --text_column content 
````

To make just a test run with a subset of the data set `test_run` argument to True.

The first time you load the dataset might be slow if it is large, but the data is saved in the cache thanks to `datasets`, and the subsequent calls will be fast.

### Alternative Deduplication Script

`minhash_deduplication_alt.py` is an alternative you might find useful to use as well. It is best for a single multi-core machine environment and uses similar parameters to the original deduplication script.

```bash
pip install -r requirements_alt.txt
# For details on the arguments, see the help message
python minhash_deduplication_alt.py --help
```

#### Scaling
This script basically completes the deduplication in 4 steps:
1. Compute the minhashes for all the files in the dataset. Hashing scales with both the number of cores and single core performance (clock speed, for example). With `datasets`'s caching, it also does not require much memory.
2. Index the minhashes. Indexing is one bottleneck that ties to single core performance. It is basically putting data in to a dictionary, so it is very hard to parallelize. With a database backend(Redis/Cassandra), you can utilize multi-threads, but it will slow down the query significantly.
3. Query the index. Querying can scale with multiple cores, but it requires the system to support copy-on-write forking mechanism.
4. Cluster duplicate minhashes and post-process the results. Building a graph (slow) and finding connected components (fast) is another bottleneck that ties to single core performance (networkit uses OpenMP for some algorithms, but not all of them).

Some stats for reference: It took about 6 hours to run this script for the python permissive license dataset on a 80-core 1.8TB RAM machine.




