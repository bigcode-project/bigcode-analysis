#!/bin/bash
declare -a thresholds=(0.65 0.85)
declare -a ngrams=(1 5)
sample=1000

# for ngram_size in "${ngrams[@]}"
# do
#   for i in "${thresholds[@]}"
#   do
#     python minhash_deduplication_alt.py --dataset "codeparrot/codeparrot-clean-valid" --cache-dir ".cache"  --ngram-size ${ngram_size}  --threshold "$i"
#   done
#   python analyze.py --results-dir "results/codeparrot/codeparrot-clean-valid/default" --output ${ngram_size}-gram.png --sample-size ${sample}
# done

for ngram_size in "${ngrams[@]}"
do
  for i in "${thresholds[@]}"
  do
    python get_clusters.py --dataset-root "results/codeparrot/codeparrot-clean-valid/default"  --threshold "$i" --ngram-size ${ngram_size}
  done
  python analyze.py --results-dir "results/codeparrot/codeparrot-clean-valid/default" --output ${ngram_size}-gram-org.png --sample-size ${sample} --alternative-graph
done