# Near deduplication

Code for data near deduplication.

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
