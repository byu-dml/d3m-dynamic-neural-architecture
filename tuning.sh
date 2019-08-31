#!/bin/bash

use_complete_data=false
objective=ndcg

if $use_complete_data; then
    # complete has 194 datasets
    raw_data_path=./data/complete_classification.tar.xz
    train_path=./data/complete_classification_train.json
    test_size=44
    test_split_seed=3746673648
    validation_size=25
    validation_split_seed=3101978347
    k=25
    metafeature_subset=all
    results_dir=./results
    tuning_output_dir=./tuning_output

else
    # small has 11 datasets
    raw_data_path=./data/small_classification.tar.xz
    train_path=./data/small_classification_train.json
    test_size=2
    test_split_seed=9232859745
    validation_size=2
    validation_split_seed=5460650386
    k=2
    metafeature_subset=all
    results_dir=./dev_results
    tuning_output_dir=./dev_tuning_output
fi


python3 -m dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed


model=lstm
n_generations=2
population_size=2


python3 -m dna tune \
    --model $model \
    --model-config-path ./model_configs/${model}_config.json \
    --tuning-config-path ./tuning_configs/${model}_tuning_config.json \
    --tuning-output-dir $tuning_output_dir \
    --problem regression rank \
    --objective $objective \
    --train-path $train_path \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --model-seed $validation_split_seed \
    --output-dir $results_dir \
    --n-generations $n_generations \
    --population-size $population_size \
    --verbose
