import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='results')
args = parser.parse_args()

results_dir = args.results_dir
dirs = os.listdir(results_dir)

for dir_ in dirs:
    results_path = os.path.join(results_dir, dir_, 'run.json')
    with open(results_path) as f:
        results = json.load(f)
    id_ = results['id']
    try:
        print('ID:', id_)
        problem_scores = results['scores']
        for scores in problem_scores:
            problem_name = scores['problem_name']
            train_scores = scores['train_scores']
            test_scores = scores['test_scores']
            train_scores = train_scores['mean_scores'] if problem_name != 'subset' else train_scores
            test_scores = test_scores['mean_scores'] if problem_name != 'subset' else test_scores
            print('PROBLEM:', problem_name)
            print('MEAN TRAIN SCORES:', train_scores)
            print('MEAN TEST SCORES:', test_scores)
            print()
    except(KeyError, TypeError):
        print('Results at id: {0} do not have scores'.format(id_))
        print()
    print('#######################################################')
    print()
