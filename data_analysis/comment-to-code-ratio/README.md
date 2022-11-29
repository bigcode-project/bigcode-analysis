# Filtering based on comment to code ratio

Here we are interested in filtering files based on their comments to code ratio. We can expect files with a higher number of comments and docstrings to be of better quality.On the other hand files where the majority of lines are comments may not be as uselful for a code generation model. We filter with a minimum and maximum comment to code ratio, which is computed in the following way:
    * For Python, we extract comments using Python tokenizer and docstrings using `ast` parsing.
    * For other languages (Java and Javascript), we extract comments using `pygments` library.
    * We compute the comment to code ratio of a file by counting the number of characters in comments over the total number of characters in the file.

You can find clean filtering code in `bigcode-dataset`repository under [preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing).
* `analysis_comments_ratio.ipynb` contains the code for the analysis of the comment to code ratio filter, used to come up with minimum and maximum thresholds (0.01 and 0.8) for the Python, Java and JavaScript subsets of [The Stack](https://huggingface.co/datasets/bigcode/the-stack).