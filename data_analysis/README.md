# Data anlaysis

In this folder we provide code for analysis of code datasets:
* Near deduplication using MinHash and LSH

* Data decontamination from HumanEval and MBPP evaluation benchmarks

* Python data analysis:
    * Natural language distribution in comments/docstrings 
    * Detection of configuration and test files (valid for other languages than Python)
    * Estimation of the number of files that can be successfully compiled

* Comment to code ratio: analysis notebook for filtering based on the ratio of comments in a file. Filtering code available at [bigcode-dataset/preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing)

* Stars filtering: analysis notebook for filtering based on the number of stars of files. Filtering code available at [bigcode-dataset/preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing)

* PII Redaction: moved to [bigcode-dataset/preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/pii)
    * PII detection of emails, IP addresses and secret keys
    * PII anonymization
    * Pipeline evaluation on an annotated benchmark

* Preprocessing:  moved to [bigcode-dataset/preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing)
   * code for data filtering based on line length and percentage of alphanumeric characters, comment to code ratio and stars.

