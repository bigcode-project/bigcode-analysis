# Natural Language identification in Python code

In this folder, we provide code to extract Python docstrings and comment and identify their natural language.

# Setup
We use `fasttext` for language identification, download the language detection model `lid.176.bin` from [fasttext.cc/docs/en/language-identification](https://fasttext.cc/docs/en/language-identification.html) and seve it in `fastext_model`folder. You need to install `fastext` and `datasets` libraries.

```
pip install fastext
pip install datasets
```

# Usage
The command below saves a dataset with additional columns giving the language of each file, the score/confidence of model in the prediction, the extracted natural text and its size:
````
python language_identifier.py \
    --dataset_name <DATA>\
    --model_path fasttext_model/lid.176.bin\
    --save_path ./data/
````
# Analysis

See the notebook `analysis.ipynb`.

Detected language distribution on 2,000 samples from CodeParrot data:
<h3 align="center">
    <img width="360" height="300" src="https://user-images.githubusercontent.com/44069155/191994477-6246467b-eec7-4ae1-a14d-dd2262254762.png" /></a>
</h3>
