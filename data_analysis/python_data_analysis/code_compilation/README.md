# Code compilation
Here we provide code to estimate the number of valid Python files by using the \texttt{py\_compile} module on some samples from a code dataset. We try to compile files for both python2 and python3 and count how many throw syntax errors.

You can execute the code using:
```bash
python compile.py --dataset_name <dataset_name> --n_samples <n_samples> --seed <seed> --text_column <text_column>
```
where `dataset_name` is the name of the dataset you want to analyze, `n_samples` is the number of samples to use, `seed` is the seed for the random shuffling.
