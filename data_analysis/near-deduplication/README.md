# Near deduplication
## For the lastest version of near-deduplication with speed-ups, check [bigcode-dataset/near_deduplication](https://github.com/bigcode-project/bigcode-dataset/tree/main/near_deduplication)
Code for running near-deduplication with MinHash and LSH indexing

### Setup

````
pip install -r requirements.txt
````

Login to be able to push the dataset to the hub after deduplication and clone your huggingface-hub repositories:

````
huggingface-cli login
````

And make sure you have git-lfs installed.

If you use datasets with different column names from the BigCode ones, you might need to change `PATH_COLUMN` and `CONTENT` variables in `minhash_deduplication.py`.

### Usage

To run near deduplication use the following command and adapt the arguments for your case:

````
python near_deduplicate.py \
    --dataset_name bigcode-data/python_any_license_v2 \
    --org bigcode-data \
    --repo_name python_any_license_v2_near_dedup \
    --out_path ./data/any_license-near-dedup \
    --text_column content 
````

To make just a test run with a subset of the data, set `test_run` argument to True.

The first time you load the dataset might be slow if it is large, but the data is saved in the cache thanks to `datasets`, and the subsequent calls will be fast.

### Alternative Deduplication Script

`minhash_deduplication_alt.py` is an alternative you might find useful to use as well. It is best for a single multi-core machine environment and uses similar parameters to the original deduplication script.

```bash
pip install -r requirements_alt.txt
# Quick example
python minhash_deduplication_alt.py --dataset codeparrot/codeparrot-clean-valid \  
    --split train \
    --column content \
    --cache-dir .cache \
    --verbose
# For details on the arguments, see the help message
python minhash_deduplication_alt.py --help
```

#### Implementation Analysis

This is for the alternative script that is designed for single-machine setup.

##### Scaling

To understand the limitation of current deduplication implementation, it is important to have an idea of how each step in the pipeline affects the overall time:
1. Minhashing is fast, but it takes longer for long documents. Hashing scales with both the number of cores and single core performance (clock speed, for example). With `datasets`s caching, it also does not require much memory.
2. Indexing is basically putting minhash signatures into different buckets. This is one bottleneck in this pipeline. In an ideal situation where MapReduce is seamlessly integrated with other parts, it can be further improved with distributed buckets.
3. Depending on how you look at duplicates, querying can be easily created by iterating the buckets or iterating the simhashes.
4. Depending on how you decide to group duplicates, you can build a graph and then do connected component analysis or use a simple algorithm like union-find.
5. What to do with a group of duplicates is also a widely open question. We opt to keep one document within a group/cluster in this case.

##### Experiments

We report here some stats on the experiments we did along the way with a 80-core machine on GCP (M1):

For SantaCoder, our results can be replicated by the following commands:

```bash
python minhash_deduplication_alt.py --dataset bigcode/the-stack-dedup-pjj --data-dir data/java --revision v1.1.a1 --cache-dir cache2 --ngram-size 5 --threshold 0.7 --min-token-length 10 --fast
python minhash_deduplication_alt.py --dataset bigcode/the-stack-dedup-pjj --data-dir data/javascript --revision v1.1.a1 --cache-dir cache2 --ngram-size 5 --threshold 0.7 --min-token-length 10 --fast
python minhash_deduplication_alt.py --dataset bigcode/the-stack-dedup-pjj --data-dir data/python --revision v1.1.a1 --cache-dir cache2 --ngram-size 5 --threshold 0.7 --min-token-length 10 --fast
```

Java Results as of Dec 20, 2022
```
load_dataset                    : 3414.68 seconds
minhash                         : 22966.13 seconds
clustering                      : 7676.72 seconds
filtering                       : 1118.62 seconds
save                            : 3105.66 seconds
Data Number (before)            : 40113161
Data Number (after)             : 21108567 (52.62%)
Duplicate Number                : 19004594 (47.38%)
Total Time                      : 38281.88 seconds (10.6 hours)
```


Java (already deduplicated) Results as of Dec 2, 2022
```
Load Dataset                    : 77.18 seconds                                                                                       
Embed                           : 5052.87 seconds                                                                                     
Create Index                    : 16253.12 seconds                                                                                    
Save Index                      : 0.00 seconds                                                                                        
Freeze Memory                   : 0.00 seconds                                                                                        
Query                           : 1321.61 seconds                                                                                     
Save Neighbors                  : 0.00 seconds                                                                                        
Unfreeze Memory                 : 0.00 seconds                                                                                        
Clustering                      : 10825.30 seconds                                                                                    
Total Processing Time           : 34919.87 seconds                                                                                    
Deduplicate                     : 605.83 seconds                                                                                      
Save Deduplicated               : 2356.10 seconds                                                                                     
Language                        : java                                                                                                
Data Number (before filtering)  : 25124914                                                                                            
Data Number (after filtering)   : 24972491                                                                                            
Duplicate Number                : 4822205 (19.31%)                                                                                    
Total Reduction                 : 4974628 (19.80%)                                                                                    
Total Time                      : 37881.83 seconds (10.5 hours)                                                                        
```

More details can be found on https://zippy-anise-556.notion.site/Deduplication-Log-d75d1b3f2e684e96a12b069c5aff68cb.
