#!/bin/bash

# actual train test split - DO NOT CHANGE
# complete has 194 datasets; small has 11 datasets
raw_data_path=./data/small_classification.tar.xz  #  ./data/complete_classification.tar.xz
test_size=2  # 44
test_split_seed=9232859745  # 3746673648


train_path=./data/small_classification_train.json
validation_size=2  # 25
validation_split_seed=5460650386
k=2
results_dir=./dev_results


python3 dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed


python3 dna evaluate \
    --model autosklearn \
    --problem rank \
    --k $k \
    --scores top-k-count top-1-regret \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose


python3 dna evaluate \
    --model mean_regression \
    --problem regression \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose


python3 dna evaluate \
    --model median_regression \
    --problem regression \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose


python3 dna evaluate \
    --model per_primitive_regression \
    --problem regression \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose


python3 dna evaluate \
    --model dna_regression \
    --model-config-path ./model_configs/dna_regression_config.json \
    --problem regression rank \
    --k $k \
    --scores top-k-count top-1-regret spearman \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose
