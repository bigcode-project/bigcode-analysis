# PII detection and redaction for Emails, IP adresses and Secret keys

We provide code to detect Emails, IP addresses and API/SSH keys in text datasets (in particular datasets of source code). We use regexes for emails and IP addresses (they are adapted from [BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii)). And we use [detect-secrets](https://github.com/Yelp/detect-secrets) for finding secrets keys. We additionally implement some filters on top to reduce the number of false positives. There is also some evaluation code to test the pipeline on a PII benchamrk we annotated.

## Usage
```
pip install -r requirements.txt
```

* `main.py` is the main script to run the pipeline. It takes as input a dataset and outputs a new dataset with the PII removed and some additional column containing the secrets found and their statistics.

```
python main.py --dataset_name <DATASET_NAME> --batch_size <BATCH_SIZE> --num_proc <NUMP_PROC> --target_dataset <TARGET_DS_NAME>
```
Make sure you have the `gibberish_data` folder in the same directory as the script. It contains a [gibberish-detector](https://github.com/domanchi/gibberish-detector) that we use for the filters for keys.

* `pii_detection.py` contains the code to perform PII detection.
* `pii_redaction.py` contains the code to redact the PII.
*  `utils/evaluation.py` contains the code to evaluate the PII detection on our annotated benchmark, with `tests` containing some test cases. (TODO: add script for automatic evaluation on the benchmark)

## Notebooks
* `example.ipynb` is an example notebook to show how to use the pipeline.
* `quantitative_evaluation_regexes.ipynb` contains some evaluation the PII detection.
* there are several notebooks in `notebooks` folder with some of our experiments.