#!/bin/bash
# This script is used to run the near-deduplication analysis.
declare -a thresholds=(0.6 0.65 0.7 0.75 0.8 0.85)
declare -a ngrams=(1 5)
sample=1000
dataset="bigcode/the-stack-pjjs-medium"

for ngram_size in "${ngrams[@]}"
do
  for i in "${thresholds[@]}"
  do
    python minhash_deduplication_alt.py --dataset "${dataset}" --data-dir "data/python" --cache-dir ".cache"  --ngram-size ${ngram_size}  --threshold "$i"
  done
  python false_positive.py --results-dir "results/${dataset}/default" --output ${ngram_size}-gram.png --sample-size ${sample}
done

for ngram_size in "${ngrams[@]}"
do
  for i in "${thresholds[@]}"
  do
    python get_clusters.py --dataset-root "results/${dataset}/default"  --threshold "$i" --ngram-size ${ngram_size}
  done
  python false_positive.py --results-dir "results/${dataset}/default" --output ${ngram_size}-gram-org.png --sample-size ${sample} --alternative-graph
done
