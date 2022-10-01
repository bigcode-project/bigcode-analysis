# PII

This code is adapted from [BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii). Please refer to this dataset [code_pii_data](https://huggingface.co/datasets/loubnabnl/code_pii_data) resulting frol running this PII on 500 samples of python programs. Detected information is in the column `regex_metadata`.

We detect email adresses , IP adresses and some keys and hide them in the data. In code datasets, one needs to be careful to not remove data that is not sensitive information but have similar format, python version (python@2.7) can be mistaken for an email, a program for encoding/hashing strings for examples can be mistaken for including keys. User detection can mistake decorators for usernames.

* emails with domain and at least 3 characters before "." are detected and replaced with dummy@email.co
* IP adresses are detected and replaced with localhost adress 127.0.0.1
* long keys with a long patterns (ssh, api) are detected and replaced by PI:Key

`pii_processor`executes PII on a dataset and saves result. Scripts for running jobs on multiple datasets (from BigScience) are available in `script`.