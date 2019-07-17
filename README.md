# Dynamic Neural Architecture (DNA)
This repository contains a system for evaluating metalearning systems on a meta-dataset (a dataset containing info about machine learning experiments on datasets).  Models range from the simple (random, linear regression) to the complex (deep neural networks, AutoSKLearn) and are evaluated on numerous metric (see `dna/metrics.py`).

## Instructions for use:
0. Setup the python enviroment (`python3 install -r requirements.txt`)
1. The command `bash main.sh` will run all the models available with the configuration shown in `__main__.py`.  To run one model, run a command similar to the ones being run in `main.sh`.

## Configuration and results
2. To edit the models being run, edit `main.sh` (options are located in `__main__.py`)
3. The full dataset will be available at this link: `link_no_yet_ready`.  Place it next to the small dataset at `data/complete_classification.tar.xz`, uncommenting out lines 3-10 and commenting out lines 13-20.
4. Results are found in `dev_results/_name_of_your_model_run_/` where `_name_of_your_model_run_` can be found printed out at the top of the `bash main.sh` command.

## How to contribute a new model:
0. Add your new model code to `dna/models/_your_model_name_.py`.  It should inherit from the base classes of the tasks it can perform (RegressionBase, RankingBase, SubsetBase).  
1. Please add tests to `tests/test_models.py`.
1. Once the model inherits from those classes and overrides their methods, the model should be added to the list found in the function `get_models` of the file `dna/models.py`.  You can then run your model from the command line, or by adding it to `main.sh`

## How to contribute a new metric:
0. Add your metric code to `dna/metrics.py`. 
1. Please add tests to the file `tests/test_metrics.py`.
2. Add your metric to the code found in `dna/problems.py` under the respective task the metric goes under (for example, the spearman metric is found under `RankProblem`). Add the results to the json dictionary returned from the function.
3. You can see your metric in action by running a model from the command line, or by running `main.sh`. 


