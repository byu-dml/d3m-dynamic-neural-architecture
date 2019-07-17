#!/bin/bash

# # complete has 194 datasets
# raw_data_path=./data/complete_classification.tar.xz
# train_path=./data/complete_classification_train.json
# test_size=44
# test_split_seed=3746673648
# validation_size=25
# validation_split_seed=3101978347
# k=25
# metafeature_subset=all


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


python3 -m dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed 


python3 -m dna evaluate \
    --model autosklearn \
    --problem regression rank subset \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model mean_regression \
    --problem regression \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines



python3 -m dna evaluate \
    --model median_regression \
    --problem regression \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines



python3 -m dna evaluate \
    --model per_primitive_regression \
    --problem regression rank subset \
    --k $k \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model linear_regression \
    --problem regression rank subset\
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model meta_autosklearn \
    --model-config-path ./model_configs/meta_autosklearn_config.json \
    --problem regression rank subset \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model random \
    --problem rank subset \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model dna_regression \
    --model-config-path ./model_configs/dna_regression_config.json \
    --problem regression rank subset \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model daglstm_regression \
    --model-config-path ./model_configs/daglstm_regression_config.json \
    --problem regression rank subset \
    --k $k \
    --metafeature-subset $metafeature_subset \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model hidden_daglstm_regression \
    --model-config-path ./model_configs/hidden_daglstm_regression_config.json \
    --problem regression rank subset \
    --k $k \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines


python3 -m dna evaluate \
    --model probabilistic_matrix_factorization \
    --model-config-path ./model_configs/probabilistic_matrix_factorization_config.json \
    --problem regression rank subset \
    --k $k \
    --train-path $train_path \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --output-dir $results_dir \
    --verbose \
    --out-of-training-set-pipelines
