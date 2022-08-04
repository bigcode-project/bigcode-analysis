
# BigCode data analysis
This repository is for the analysis of the code datasets used in BigCode. We are particularly intrested in these two datasets: [python-all-license](https://huggingface.co/datasets/BigCode/github_dump_python_only_any_license_decompressed) and [python-safe-license](https://huggingface.co/datasets/BigCode/github_dump_v2_python_only_safe_licenses).

### Notebooks
You can find 3 notebooks for analyzing the loss of the models trained on these datasets, the file size distribution and loss analysis through clustering.

### Additional analysis:
* Filtering:

The filtering of these datasets based on the number of configuration and test files present as well as other cleaning filters, removes 33.58% of the files and 20% of the data in python-all-license, and 34% of the files and 22% of the data in volume from python-safe-license. Filtered datasets are available here: [python-all-license-filtered](https://huggingface.co/datasets/BigCode/github-python-all-license-conf-test-filter) and [python-safe-license-filtered](https://huggingface.co/datasets/BigCode/github-python-safe-license-conf-test-filter).

* Near deduplication:

Near deduplication removes 27% of the files and 47% of the python-all-license dataset in volume. The near deduplicated version is available [here](https://huggingface.co/datasets/BigCode/github-python-any-license-near-dedup). (Near deduplication of python-safe-license is  ongoing).
