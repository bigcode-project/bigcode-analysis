# Decontamination

This directory contains several scripts for decontamination of the data.
1. Exact prompt matching
2. Near matching

## Near Matching with MinHash and LSH

This is similar to the near deduplication script `data_analysis/near-deduplication/minhash_deduplication_alt.py` with one modification: we use benchmark datasets as index source instead of the dataset itself.

### Usage:
1. Update the script to include any benchmark you want to check agains in `DATASETS_TO_CHECK`. Be sure to create a global variable for the index using the same name in that config. Benchmark columns should be of type string or sequence of string, so that they can be concatenated.
2. Then you can run the script by
```bash
pip install -r requirements_minhash.txt
# Quick example
python minhash.py \
  --dataset codeparrot/codeparrot-clean-valid \
  --split train \
  --column content \
  --cache-dir .cache \
  --verbose
# Check parameters with the help message
python minhash.py --help
```