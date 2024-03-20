# Scripts for preference model pretraining data

### Stack Exchange
Note: Stack Exchange Data Dump has a license requiring the addition of author's and links to the original material, see more [here](https://archive.org/details/stackexchange).

1) `stack_exchange_explore.py`: example script for filtering stack exchange data to the question & answer format in Askell et al. 2021 on preference model pretraining (PMP).

To run this code (from scratch including data download and faster processing), do the following:
Identify the raw data directory you're hoping to process, `ex_data_url`, and related data variables (further string optimizations can be added).
The script will pull raw data if you need it, uncompress it, and process the file to text.

```shell
python scripts/data/pmp/stack_exchange_explore.py --stack_exchange=pets --save=True
```

2) `stack_exchange_process.py`: same as above, but designed to be run on a large machine to process all files consecutively.
It is a long for-loop over desired exchanges.

```shell
python scripts/data/pmp/stack_exchange_process.py --save_path=/path/to/hf-dataset
```

3) `binarize.py`: used to binarize the pre-filter Stack Exchange data
```shell
python scripts/data/pmp/binarize.py --save_path=/path/to/hf-dataset
```

Credits: code from HuggingFaceH4 team
