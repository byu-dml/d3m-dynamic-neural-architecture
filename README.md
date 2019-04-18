# d3m-pipeline-regressor


## Instructions for use:
0. Copy the compressed data (<filename>.tar.xz) into ./data
1. Configure the dataset path and size of the test split in ./dna/data/py
2. Extract the data and create train/test splits by running `python3 dna/data.py`
3. Configure the model type (`'regression'` or `'siamese'`) by setting `task` in ./dna/main.py
4. Run the model using `python3 dna/main.py`
