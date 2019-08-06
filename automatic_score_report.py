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
        scores = results['scores'][0]
        train_scores = scores['train_scores']
        test_scores = scores['test_scores']
        print('ID:', results['id'])
        print('MEAN TRAIN SCORES:', train_scores['mean_scores'])
        print('MEAN TEST SCORES:', test_scores['mean_scores'])
    except(KeyError, TypeError):
        print('Results at id: {0} do not have scores'.format(id_))
    print()
