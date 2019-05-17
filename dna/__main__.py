import argparse
import json
import os
import sys
import typing

from data import get_data, preprocess_data, split_data


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
        '--test-size', type=int, action='store', default=40,
        help='the number of datasets in the test split'
    )
    parser.add_argument(
        '--split-seed', type=int, action='store', default=3746673648,
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
        train_path = os.path.join(dirname, 'train_' + data_filename + '.json')

    test_path = getattr(arguments, 'test_path')
    if test_path is None:
        dirname, data_filename = data_path.rsplit(os.path.sep, 1)
        data_filename, ext = data_filename.split('.', 1)
        test_path = os.path.join(dirname, 'test_' + data_filename + '.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f)

    with open(test_path, 'w') as f:
        json.dump(test_data, f)


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
        '--problem', type=str, action='store', required=True,
        choices=['regression', 'siamese'],
        help='the type of problem'
    )
    parser.add_argument(
        '--model', type=str, action='store', required=True,
        help='the python path to the model class'
    )
    parser.add_argument(
        '--model-config-path', type=str, default=None,
        help='path to a json file containing the model configuration values'
    )


def evaluate_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    data_resolver=get_data
):
    train_path = getattr(arguments, 'train_path')
    train_data = data_resolver(train_path)

    if getattr(arguments, 'test_path') is None:
        train_data, test_data = split_data(
            train_data, 'dataset_id', getattr(arguments, 'test_size'),
            getattr(arguments, 'split_seed')
        )
    else:
        test_path = getattr(arguments, 'test_path')
        test_data = data_resolver(test_path)

    train_data, test_data = preprocess_data(train_data, test_data)


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser):
    subparser = parser._subparsers._group_actions[0].choices[arguments.command]

    if arguments.command == 'split-data':
        split_handler(arguments, subparser)

    elif arguments.command == 'evaluate':
        evaluate_handler(arguments, subparser)

    else:
        raise ValueError('A suitable command handler could not be found.')


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
