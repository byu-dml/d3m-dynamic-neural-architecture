#!/bin/bash

# # complete has 194 datasets
# raw_data_path=./data/complete_classification.tar.xz
# train_path=./data/complete_classification_train.json
# test_size=44
# test_split_seed=3746673648
# validation_size=25
# validation_split_seed=3101978347
# k=25

# small has 11 datasets
raw_data_path=./data/small_classification.tar.xz
train_path=./data/small_classification_train.json
test_size=2
test_split_seed=9232859745
validation_size=2
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
    --scores top-k-count top-1-regret top-k-regret \
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
    --scores top-k-count top-1-regret spearman top-k-regret pearsons_correlation \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose

python3 dna evaluate \
    --model dagrnn_regression \
    --model-config-path ./model_configs/dagrnn_regression_config.json \
    --problem regression rank \
    --k $k \
    --scores top-k-count top-1-regret spearman top-k-regret \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose
