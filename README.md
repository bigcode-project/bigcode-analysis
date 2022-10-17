
# BigCode Analysis
This repository is for the analysis done in BigCode Project. You can find analysis of datasets, models, architecture choices and more.

## Contents
* **Data analysis**: In the folder `data_analysis`, we analyze these two datasets: python-all-license (private) and [python-safe-license](https://huggingface.co/datasets/BigCode/github_dump_v2_python_only_safe_licenses). We provide the following statistics:
  * percentage of near duplicates
  * percentage of configuration/test and uncommon files 
  * file size distribution
  * loss analysis
  * natural language distribution in comments/docstrings and number of files that can be successfully compiled
  
We also provide code to run near-deduplication, and to detect natural language of comments in Python datasets.

* **Multi query attention experiments**, for details refer [here](/multi_query_experiments/README.md)
