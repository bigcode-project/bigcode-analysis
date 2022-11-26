#!/bin/bash
declare -a thresholds=(0.6 0.65 0.7 0.75 0.8 0.85)
declare -a ngrams=(1 3 5)

for ngram_size in "${ngrams[@]}"
do
  for i in "${thresholds[@]}"
  do 
   python minhash_deduplication_alt.py --dataset "bigcode/the-stack-dedup-pjj" --data-dir "data/python" --cache-dir "cache2" --local-dir "cache2/the-stack-dedup-pjj-p-tiny" --ngram-size ${ngram_size}  --threshold "$i" 
  done
  python analyze.py --results-dir "results/bigcode/the-stack-dedup-pjj/default" --output ${ngram_size}-gram.png --sample-size 1000
done