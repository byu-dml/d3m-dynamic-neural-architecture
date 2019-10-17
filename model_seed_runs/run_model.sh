MODEL_CONFIG_PATH=./model_configs/${MODEL}_config.json
OUTPUT_DIR=./model_seed_runs/${MODEL}_results

for i in {1..10};
do
	python3 -m dna evaluate \
	--model $MODEL \
	--model-config-path $MODEL_CONFIG_PATH \
	--problem regression rank \
	--train-path ./data/complete_classification_train.json \
	--test-size 25 \
	--split-seed 3101978347 \
	--output-dir $OUTPUT_DIR \
	--verbose
done
