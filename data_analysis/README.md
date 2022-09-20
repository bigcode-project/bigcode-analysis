
# Data analysis
This folder is intended for data analysis, we are interested in these two datasets: [python-all-license](https://huggingface.co/datasets/BigCode/github_dump_python_only_any_license_decompressed) and [python-safe-license](https://huggingface.co/datasets/BigCode/github_dump_v2_python_only_safe_licenses).
For filtering and near deduplication we use the preprocessing code in [CodeParrot project](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot)
### Notebooks
You can find 3 notebooks for analyzing the loss of the models trained on these datasets, the file size distribution and loss analysis through clustering.

### Additional analysis:
* Filtering:

The filtering of these datasets based on the number of configuration and test files present as well as other cleaning filters, removes 33.58% of the files and 20% of the data in python-all-license, and 34% of the files and 22% of the data in volume from python-safe-license. Filtered datasets are available here: [python-all-license-filtered](https://huggingface.co/datasets/BigCode/github-python-all-license-conf-test-filter) and [python-safe-license-filtered](https://huggingface.co/datasets/BigCode/github-python-safe-license-conf-test-filter).

* Near deduplication:

Near deduplication removes 27% of the files and 47% of the python-all-license dataset in volume. The near deduplicated version is available [here](https://huggingface.co/datasets/BigCode/github-python-any-license-near-dedup). 

Near deduplication removes 36% of the files and 58% of the dataset in volume of the python-all-license dataset in volume. The near deduplicated version is available [here](https://huggingface.co/datasets/BigCode/github-python-safe-licenses-near-dedup). Below are the statistics of the data removed by each filtering.

|dataset | size of duplicates | number of duplicate files |size filtered files (*)|
|-------|--------|---------|---------|
|safe licenses (179GB, 23M files)| 103GB (**58%**)| 8.4M (**36%**)| 39GB (**22%**)|
|all licenses (234GB, 42M files)|109GB (47%)| 11.6M (27%)| 46GB (20%)|

(*) config/test/uncommon

File size distribution of random 10k subsets (before and after filtering):

<img width="493" alt="image" src="https://user-images.githubusercontent.com/44069155/183675926-13bf6f5b-b9c7-4def-91fb-4fb46ee1d505.png">

<img width="493" alt="image" src="https://user-images.githubusercontent.com/44069155/183676271-962dff8b-4759-4950-b8cb-f07d2364ebde.png">
