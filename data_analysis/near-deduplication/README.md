# Near deduplication

Code for running near-deduplication with MinHash and LSH indexing

### Setup

````
pip install -r requirements.txt
````

Login to be able to be able to push the dataset to the hub after deduplication and clone your huggingface-hub repositories:

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

To make just a test run with a subset of the data set `test_run` argument to True.

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

To understand the limitation of current deduplication implementation, it is important to have an idea of how each step in the pipleine affects the overall time:
1. Minhashing is fast, but it takes loner for long documents. Hashing scales with both the number of cores and single core performance (clock speed, for example). With `datasets`'s caching, it also does not require much memory.
2. Indexing is basically putting minhash signatures into different buckets. This is one bottleneck in this pipleine. In an ideal situation where MapReduce is seamlessly integrated with other parts, it can be further improved with distributed buckets.
3. Depending on how you look at duplicates, querying can be easily created by iterating the buckets or iterating the simhashes.
4. Depending on how you decide to group duplicates, you can build a graph and then do connected component analysis or use simple algorithm like union-find.
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
INFO:__main__:load_dataset                    : 3414.68 seconds
INFO:__main__:minhash                         : 22966.13 seconds
INFO:__main__:clustering                      : 7676.72 seconds
INFO:__main__:filtering                       : 1118.62 seconds
INFO:__main__:save                            : 3105.66 seconds
INFO:__main__:Data Number (before)            : 40113161
INFO:__main__:Data Number (after)             : 21108567 (52.62%)
INFO:__main__:Duplicate Number                : 19004594 (47.38%)
INFO:__main__:Total Time                      : 38281.88 seconds
INFO:__main__:Deduplicated Dataset            : results/output/deduplicated
INFO:__main__:🤗 Happy Deduplicating 🤗
```


Java Results as of Dec 2, 2022
```
[12/03/22 13:37:40] INFO     Load Dataset                    : 77.18 seconds                                                                                       
                    INFO     Embed                           : 5052.87 seconds                                                                                     
                    INFO     Create Index                    : 16253.12 seconds                                                                                    
                    INFO     Save Index                      : 0.00 seconds                                                                                        
                    INFO     Freeze Memory                   : 0.00 seconds                                                                                        
                    INFO     Query                           : 1321.61 seconds                                                                                     
                    INFO     Save Neighbors                  : 0.00 seconds                                                                                        
                    INFO     Unfreeze Memory                 : 0.00 seconds                                                                                        
                    INFO     Clustering                      : 10825.30 seconds                                                                                    
                    INFO     Total Processing Time           : 34919.87 seconds                                                                                    
                    INFO     Deduplicate                     : 605.83 seconds                                                                                      
                    INFO     Save Deduplicated               : 2356.10 seconds                                                                                     
                    INFO     Language                        : java                                                                                                
                    INFO     Data Number (before filtering)  : 25124914                                                                                            
                    INFO     Data Number (after filtering)   : 24972491                                                                                            
                    INFO     Duplicate Number                : 4822205 (19.31%)                                                                                    
                    INFO     Total Reduction                 : 4974628 (19.80%)                                                                                    
                    INFO     Total Time                      : 37881.83 seconds
```

More details can be found on https://zippy-anise-556.notion.site/Deduplication-Log-d75d1b3f2e684e96a12b069c5aff68cb.