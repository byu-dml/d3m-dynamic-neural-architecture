#!/bin/bash

python3 dna evaluate \
    --train-path ./data/train_small_classification.json \
    --test-size 2 \
    --split-seed 8520 \
    --problem rank \
    --k 25 \
    --model autosklearn \
    --scores top-k-count top-1-regret \
    --output-dir ./results \
    --verbose

python3 dna evaluate \
    --train-path ./data/train_small_classification.json \
    --test-size 2 \
    --split-seed 8520 \
    --problem regression rank \
    --k 25 \
    --model dna_regression \
    --model-config-path ./model_configs/dna_regression_config.json \
    --model-seed 866282856 \
    --scores top-k-count top-1-regret spearman \
    --output-dir ./results \
    --verbose

python3 dna evaluate \
    --train-path ./data/train_small_classification.json \
    --test-size 2 \
    --split-seed 8520 \
    --problem regression \
    --model mean_regression \
    --model-seed 866282856 \
    --output-dir ./results \
    --verbose

python3 dna evaluate \
    --train-path ./data/train_small_classification.json \
    --test-size 2 \
    --split-seed 8520 \
    --problem regression \
    --model median_regression \
    --model-seed 866282856 \
    --output-dir ./results \
    --verbose

python3 dna evaluate \
    --train-path ./data/train_small_classification.json \
    --test-size 2 \
    --split-seed 8520 \
    --problem regression \
    --model per_primitive_regression \
    --model-seed 866282856 \
    --output-dir ./results \
    --verbose

# # pairwise classification model/task
# python3 dna evaluate \
#     --train-path ./data/train_small_classification.json \
#     --test-size 2 \
#     --split-seed 8520 \
#     --problem binary-classification \
#     --model dna_siamese \
#     --model-config-path ./model_configs/dna_siamese_config.json \
#     --model-seed 866282856 \
#     --output-dir ./results \
#     --verbose

# # using actual test data
# python3 dna evaluate \
#     --train-path ./data/train_small_classification.json \
#     --test-path ./data/test_small_classification.json \
#     --problem regression rank top-k \
#     --model dna_regression \
#     --model-config-path ./model_configs/dna_regression_config.json \
#     --model-seed 866282856 \
#     --output-dir ./results \
#     --verbose
