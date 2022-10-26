# PII [WIP]

This code is adapted from [BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii). For examples refer to this dataset [code_pii_data](https://huggingface.co/datasets/loubnabnl/code_pii_data) resulting from running this PII on 500 samples of python programs. Detected information is in the column `regex_metadata`.

We detect email adresses , IP adresses and some keys and hide them in the data. In code datasets, one needs to be careful to not remove data that is not sensitive information but has similar format, python version (python@2.7) can be mistaken for an email, a program for encoding/hashing strings for example can be mistaken for including keys. User detection can mistake decorators for usernames.

Currentky, we remove: 
* emails with domain name and at least 3 characters before "." are detected and replaced with dummy@email.com
* IP adresses are detected and replaced with localhost adress 127.0.0.1
* keys with a long patterns (ssh, api) are detected and replaced by PI:Key

`pii_processor.py` executes PII on a dataset and saves the result. Scripts for running jobs on multiple datasets (from BigScience) are available in `script`.
The notebook `analysis.ipynb` shows some PII detection examples with regex and Presidio.
