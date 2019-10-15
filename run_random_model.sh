#!/bin/bash

use_complete_data=false
# use_ootsp=--use-ootsp

if $use_complete_data; then
    # complete has 194 datasets
    raw_data_path=./data/complete_classification.tar.xz
    train_path=./data/complete_classification_train.json
    test_path=./data/complete_classification_test.json
    test_size=44
    test_split_seed=3746673648
    validation_size=25
    validation_split_seed=3101978347
    metafeature_subset=all
    results_dir=./final_results

else
    # small has 11 datasets
    raw_data_path=./data/small_classification.tar.xz
    train_path=./data/small_classification_train.json
    test_path=./data/small_classification_test.json
    test_size=2
    test_split_seed=9232859745
    validation_size=2
    validation_split_seed=5460650386
    metafeature_subset=all
    results_dir=./dev_random_results
fi

python3 -m dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed


python3 -m dna evaluate \
    --model random \
    --problem rank \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-path $test_path \
    --output-dir $results_dir \
    --verbose \
    $use_ootsp


python3 -m dna evaluate \
    --model random \
    --problem rank \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-path $test_path \
    --output-dir $results_dir \
    --verbose \
    $use_ootsp


python3 -m dna evaluate \
    --model random \
    --problem rank \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-path $test_path \
    --output-dir $results_dir \
    --verbose \
    $use_ootsp


python3 -m dna evaluate \
    --model random \
    --problem rank \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-path $test_path \
    --output-dir $results_dir \
    --verbose \
    $use_ootsp
