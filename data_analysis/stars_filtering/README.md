# Filtering based on number of stars

Here we are interested in filtering files based on their number of stars (i.e. of their parent repositories). 

You can find clean filtering code in `bigcode-dataset`repository under [preprocessing](https://github.com/bigcode-project/bigcode-dataset/tree/main/preprocessing).
* `stars_analysis.ipynb` contains the code for the analysis of the stars filter, used to come up with minimum threshold of 5 stars for Python, Java and JavaScript subsets of [The Stack](https://huggingface.co/datasets/bigcode/the-stack).