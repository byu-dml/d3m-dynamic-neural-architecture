#!/bin/bash

# MODEL must be set as an env var
if [ -z $MODEL ]; then
    echo "MODEL not set"
    exit 1
fi


MODE=DEV # DEV PROD
COMMAND=evaluate  # evaluate tune split-data
# evaluate args
N_RUNS=5  # with different seeds
# tune args
OBJECTIVE=ndcg_at_k
N_GENERATIONS=2
POPULATION_SIZE=2


if [ $MODE == "DEV" ]; then
    # small has 11 datasets
    raw_data_path=./data/small_classification.tar.xz
    train_path=./data/small_classification_train.json
    test_path=./data/small_classification_test.json
    test_size=2
    test_split_seed=9232859745
    validation_size=2
    validation_split_seed=5460650386

    if [ "$COMMAND" == "tune" ]; then
        results_dir=./dev_results_validation_set/$MODEL
        tuning_output_dir=./dev_tuning_output
    elif [ "$COMMAND" == "evaluate" ]; then
        results_dir=./dev_results_test_set/$MODEL
    fi

elif [ "$MODE" == "PROD" ]; then
    # complete has 194 datasets
    raw_data_path=./data/complete_classification.tar.xz
    train_path=./data/complete_classification_train.json
    test_path=./data/complete_classification_test.json
    test_size=44
    test_split_seed=3746673648
    validation_size=25
    validation_split_seed=3101978347

    if [ "$COMMAND" == "tune" ]; then
        results_dir=./results_validation_set/$MODEL
        tuning_output_dir=./tuning_output
    elif [ "$COMMAND" == "evaluate" ]; then
        results_dir=./results_test_set/$MODEL
    fi
fi


if [ "$COMMAND" == "split-data" ]; then
    python3 -m dna $COMMAND \
        --data-path $raw_data_path \
        --test-size $test_size \
        --split-seed $test_split_seed

elif [ "$COMMAND" == "tune" ]; then
    python3 -m dna $COMMAND \
        --model $MODEL \
        --model-config-path ./model_configs/${MODEL}_config.json \
        --tuning-config-path ./tuning_configs/${MODEL}_tuning_config.json \
        --tuning-output-dir $tuning_output_dir \
        --problem regression rank \
        --objective $OBJECTIVE \
        --train-path $train_path \
        --test-size $validation_size \
        --split-seed $validation_split_seed \
        --output-dir $results_dir \
        --n-generations $N_GENERATIONS \
        --population-size $POPULATION_SIZE \
        --verbose

elif [ "$COMMAND" == "evaluate" ]; then
    for ((i=0; i<$N_RUNS; i++)); do
        python3 -m dna $COMMAND \
            --model $MODEL \
            --model-config-path ./model_configs_tuned/${MODEL}_config.json \
            --problem regression rank \
            --train-path $train_path \
            --test-path $test_path \
            --output-dir $results_dir \
            --verbose
    done
    # todo: aggregate results here
fi
