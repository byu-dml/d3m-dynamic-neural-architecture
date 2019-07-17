import argparse
import hashlib
import json
import os
import random
import sys
import typing
import uuid

from dna.data import get_data, preprocess_data, split_data, group_json_objects
from dna.models.models import get_model, ModelBase
from dna.problems import get_problem, ProblemBase


def configure_split_parser(parser):
    parser.add_argument(
        '--data-path', type=str, action='store', required=True,
        help='path of data to split'
    )
    parser.add_argument(
        '--train-path', type=str, action='store', default=None,
        help='path to write train data'
    )
    parser.add_argument(
        '--test-path', type=str, action='store', default=None,
        help='path to write test data'
    )
    parser.add_argument(
        '--test-size', type=int, action='store', default=1,
        help='the number of datasets in the test split'
    )
    parser.add_argument(
        '--split-seed', type=int, action='store', default=0,
        help='seed used to split the data into train and test sets'
    )


def split_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *, data_resolver=get_data
):
    data_path = getattr(arguments, 'data_path')
    data = data_resolver(data_path)
    train_data, test_data = split_data(
        data, 'dataset_id', getattr(arguments, 'test_size'),
        getattr(arguments, 'split_seed')
    )

    train_path = getattr(arguments, 'train_path')
    if train_path is None:
        dirname, data_filename = data_path.rsplit(os.path.sep, 1)
        data_filename, ext = data_filename.split('.', 1)
        train_path = os.path.join(dirname, data_filename + '_train.json')

    test_path = getattr(arguments, 'test_path')
    if test_path is None:
        dirname, data_filename = data_path.rsplit(os.path.sep, 1)
        data_filename, ext = data_filename.split('.', 1)
        test_path = os.path.join(dirname, data_filename + '_test.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f, separators=(',',':'))

    with open(test_path, 'w') as f:
        json.dump(test_data, f, separators=(',',':'))


def configure_evaluate_parser(parser):
    parser.add_argument(
        '--train-path', type=str, action='store', required=True,
        help='path to read the train data'
    )
    parser.add_argument(
        '--test-path', type=str, action='store', default=None,
        help='path to read the test data; if not provided, train data will be split'
    )
    parser.add_argument(
        '--test-size', type=int, action='store', default=1,
        help='the number of datasets in the test split'
    )
    parser.add_argument(
        '--split-seed', type=int, action='store', default=0,
        help='seed used to split the data into train and test sets'
    )
    parser.add_argument(
        '--problem', nargs='+', required=True,
        choices=['regression', 'rank', 'subset'],  # TODO: 'binary-classification'],
        help='the type of problem'
    )
    parser.add_argument(
        '--k', type=int, action='store', default=10,
        help='the number of pipelines to rank'
    )
    parser.add_argument(
        '--model', type=str, action='store', required=True,
        help='the python path to the model class'
    )
    parser.add_argument(
        '--model-config-path', type=str, default=None,
        help='path to a json file containing the model configuration values'
    )
    parser.add_argument(
        '--model-seed', type=int, default=1,
        help='seed used to control the random state of the model'
    )
    parser.add_argument(
        '--verbose', default=False, action='store_true'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='directory path to write outputs for this model run'
    )
    parser.add_argument(
        '--cache-dir', type=str, default='.cache',
        help='directory path to write outputs for this model run'
    )
    parser.add_argument(
        '--no-cache', default=False, action='store_true',
        help='when set, do not use cached preprocessed data'
    )
    parser.add_argument(
        '--metafeature-subset', type=str, default='all', choices=['all', 'landmarkers', 'non-landmarkers']
    )
    parser.add_argument(
        '--use-ootsp', default=False, action='store_true',
        help='when set, enables out-of-training-set pielines (ootsp) mode. discard some pipelines from the training' +\
            ' data and evaluate the model twice: once with test data that contains only in-training-set pipelines ' +\
            'and once with only out-of-training-set pipelines'
    )
    parser.add_argument(
        '--ootsp-split-ratio', type=float, default=0.5,
        help='Used with --use-ootsp to set the ratio of pipelines that will be in the training set'
    )
    parser.add_argument(
        '--ootsp-split-seed', type=int, action='store', default=2,
        help='Seed used with --use-ootsp used to split the train and test sets into ootsp sets'
    )
    parser.add_argument(
        '--skip-test-ootsp', default=False, action='store_true',
        help='Used with --use-ootsp evaluate the model using the ootsp splits, but only the test in-training-set ' +\
            'pipelines. This is useful to compare models that cannot make predictions on ootsp'
    )


class EvaluateResult:

    def __init__(
        self, train_predictions, fit_time, train_predict_time, train_scores, test_predictions, test_predict_time,
        test_scores
    ):
        self.train_predictions = train_predictions
        self.fit_time = fit_time
        self.train_predict_time = train_predict_time
        self.train_scores = train_scores
        self.test_predictions = test_predictions
        self.test_predict_time = test_predict_time
        self.test_scores = test_scores

    def __eq__(self, other):
        result = True
        result &= self.train_predictions == other.train_predictions
        result &= self.train_scores == other.train_scores
        result &= self.test_predictions == other.test_predictions
        result &= self.test_scores == other.test_scores
        return result


def evaluate(
    problem: ProblemBase, train_data: typing.Dict, test_data: typing.Dict, model: ModelBase, model_config: typing.Dict,
    *, verbose: bool = False, model_output_dir: str = None, plot_dir: str = None
):
    train_predictions, fit_time, train_predict_time = problem.fit_predict(
        train_data, test_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
    )
    train_scores = problem.score(train_predictions, train_data)

    if plot_dir is not None:
        problem.plot(train_predictions, train_data, train_scores, os.path.join(plot_dir, 'train'))

    test_predictions, test_predict_time = problem.predict(
        test_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
    )
    test_scores = problem.score(test_predictions, test_data)

    if plot_dir is not None:
        problem.plot(test_predictions, test_data, test_scores, os.path.join(plot_dir, 'test'))

    return EvaluateResult(
        train_predictions, fit_time, train_predict_time, train_scores, test_predictions, test_predict_time, test_scores
    )


def evaluate_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    data_resolver=get_data, model_resolver=get_model,
    problem_resolver=get_problem,
):
    run_id = str(uuid.uuid4())

    model_config_path = getattr(arguments, 'model_config_path', None)
    if model_config_path is None:
        model_config = {}
    else:
        with open(model_config_path) as f:
            model_config = json.load(f)

    output_dir = arguments.output_dir
    model_output_dir = None
    plot_dir = None
    if output_dir is not None:
        output_dir = os.path.join(getattr(arguments, 'output_dir'), run_id)
        model_output_dir = os.path.join(output_dir, 'model')
        os.makedirs(model_output_dir)
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir)

        record_run(run_id, output_dir, arguments=arguments, model_config=model_config)

    train_data, test_data = get_train_and_test_data(arguments=arguments, data_resolver=data_resolver)
    if arguments.use_ootsp:
        train_data, test_data, out_of_training_test_data = get_out_of_training_set_pipeline_data(train_data, test_data, arguments)

    model_name = getattr(arguments, 'model')
    model = model_resolver(model_name, model_config, seed=getattr(arguments, 'model_seed'))

    result_scores = []
    for problem_name in getattr(arguments, 'problem'):
        if arguments.verbose:
            print('{} {} {}'.format(model_name, problem_name, run_id))
        problem = problem_resolver(problem_name, arguments)
        evaluate_result = evaluate(
            problem, train_data, test_data, model, model_config, verbose=arguments.verbose,
            model_output_dir=model_output_dir, plot_dir=plot_dir
        )
        result_scores.append({
            'problem_name': problem_name,
            'model_name': model_name,
            **evaluate_result.__dict__,
        })
        # if we have a special out-of-training-set, use that here
        if arguments.use_ootsp and not arguments.skip_test_ootsp:
            evaluate_result_out_of_training_set = evaluate(
                problem, train_data, out_of_training_test_data, model, model_config, verbose=arguments.verbose, model_output_dir=model_output_dir
            )
            result_scores.append({
                'out_of_training_set_problem_name': problem_name,
                'out_of_training_set_model_name': model_name,
                "out_of_training_set_scores": evaluate_result.__dict__,
            })
        if arguments.verbose:
            results = evaluate_result.__dict__
            del results['train_predictions']
            del results['test_predictions']
            print(json.dumps(results, indent=4))
            if arguments.use_ootsp and not arguments.skip_test_ootsp:
                results_out_of_training_set = evaluate_result_out_of_training_set.__dict__
                del results_out_of_training_set['train_predictions']
                del results_out_of_training_set['test_predictions']
                print(json.dumps({"out_of_training_set_scores": results_out_of_training_set}, indent=4), "\n")

    if output_dir is not None:
        record_run(run_id, output_dir, arguments=arguments, model_config=model_config, scores=result_scores)


def get_train_and_test_data(arguments: argparse.Namespace, data_resolver):
    data_arg_names = ['train_path', 'test_path', 'test_size', 'split_seed', 'metafeature_subset']
    data_arg_str = ''.join(str(getattr(arguments, arg)) for arg in data_arg_names)
    cache_id = hashlib.sha256(data_arg_str.encode('utf8')).hexdigest()
    cache_dir = os.path.join(arguments.cache_dir, cache_id)
    train_cache_path = os.path.join(cache_dir, 'train.json')
    test_cache_path = os.path.join(cache_dir, 'test.json')

    load_cached_data = (not arguments.no_cache) and (os.path.isdir(cache_dir))

    # determine whether to load raw or cached data
    if load_cached_data:
        in_train_path = train_cache_path
        in_test_path = test_cache_path
    else:
        in_train_path = arguments.train_path
        in_test_path = arguments.test_path

    # when loading raw data and test_path is not provided, split train into train and test data
    train_data = data_resolver(in_train_path)
    if in_test_path is None:
        assert not load_cached_data
        train_data, test_data = split_data(train_data, 'dataset_id', arguments.test_size, arguments.split_seed)
    else:
        test_data = data_resolver(in_test_path)

    if not load_cached_data:
        train_data, test_data = preprocess_data(train_data, test_data, arguments.metafeature_subset)
        if not arguments.no_cache:
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            with open(train_cache_path, 'w') as f:
                json.dump(train_data, f, separators=(',',':'))
            with open(test_cache_path, 'w') as f:
                json.dump(test_data, f, separators=(',',':'))

    return train_data, test_data


def get_out_of_training_set_pipeline_data(train_data, test_data, arguments):
    full_data = train_data + test_data
    grouped_data_indices = group_json_objects(full_data, "pipeline_id")
    groups = list(grouped_data_indices.keys())

    rnd = random.Random()
    rnd.seed(arguments.ootsp_split_seed)
    rnd.shuffle(groups)

    # make the split into in-training set pipeline ids and out-of-training-set pipeline ids
    in_training_set_pipelines = groups[:int(len(groups) * arguments.ootsp_split_ratio)]
    train_data_split = [instance for instance in train_data if instance["pipeline_id"] in in_training_set_pipelines]
    test_data_split = [instance for instance in test_data if instance["pipeline_id"] in in_training_set_pipelines]
    out_of_training_test_data = [instance for instance in train_data if instance["pipeline_id"] not in in_training_set_pipelines]
    return train_data_split, test_data_split, out_of_training_test_data 


def record_run(
    run_id: str, output_dir: str, *, arguments: argparse.Namespace, model_config: typing.Dict,
    scores: typing.Dict = None
):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, 'run.json')

    run = {
        'id': run_id,
        'arguments': arguments.__dict__,
        'model_config': model_config,
    }
    if scores is not None:
        run['scores'] = scores

    with open(path, 'w') as f:
        json.dump(run, f, indent=4, sort_keys=True)


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser):
    subparser = parser._subparsers._group_actions[0].choices[arguments.command]

    if arguments.command == 'split-data':
        split_handler(arguments, subparser)

    elif arguments.command == 'evaluate':
        evaluate_handler(arguments, subparser)

    else:
        raise ValueError('Unknown command: {}'.format(arguments.command))


def main(argv: typing.Sequence):

    parser = argparse.ArgumentParser(prog='dna')

    subparsers = parser.add_subparsers(dest='command', title='command')
    subparsers.required = True

    split_parser = subparsers.add_parser(
        'split-data', help='creates train and test splits of the data'
    )
    configure_split_parser(split_parser)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='train, score, and save a model'
    )
    configure_evaluate_parser(evaluate_parser)

    arguments = parser.parse_args(argv[1:])

    handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
