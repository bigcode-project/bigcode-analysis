# Data anlaysis

In this folder we provide code for analysis of code datasets:
* Near deduplication using MinHash and LSH

* Data decontamination from HumanEval and MBPP evaluation banchmarks

* Python data analysis:
    * Natural language distribution in comments/docstrings 
    * Detection of configuration and test files (valid for other languages than Python)
    * Estimation of the number of files that can be successfully compiled

* PII Redaction: moved to [bigcode-dataset repository](https://github.com/bigcode-project/bigcode-dataset/tree/main/pii)
    * PII detection of emails, IP addresses and secret keys
    * PII anonymization
    * Pipeline evaluation on an annotated benchmark

* Preprocessing:  moved to [bigcode-dataset repository](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing)
   * code for data filtering based on line length and percentage of alphanumeric characters, comment to code ratio and stars (see below).

* Comment to code ratio analysis

* Filtering based on stars analysis
