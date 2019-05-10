import argparse
import json
import os
import sys
import typing

from data import get_data, split_data


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
        '--seed', type=int, action='store', default=3746673648,
        help='seed used to split the data into train and test sets'
    )

def split_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *, data_resolver=get_data
):
    data_path = getattr(arguments, 'data_path')
    data = data_resolver(data_path)
    train_data, test_data = split_data(
        data, 'dataset_id', getattr(arguments, 'test_size'),
        getattr(arguments, 'seed')
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


def configure_train_parser(parser):
    parser.add_argument(
        '--train-data', type=argparse.FileType('r'), action='store',
        required=True, help='path of train data'
    )
    parser.add_argument(
        '--test-data', type=argparse.FileType('r'), action='store', default=None,
        help='path of test data'
    )
    parser.add_argument(
        '--data-seed', type=int, action='store',
        help='Random number generator seed used to split train data into train and validation sets.'
    )


def train_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
):
    print(getattr(arguments, 'train_data'))


def configure_evaluate_parser(parser):
    pass


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser):
    subparser = parser._subparsers._group_actions[0].choices[arguments.dna_command]

    if arguments.dna_command == 'split-data':
        split_handler(arguments, subparser)

    elif arguments.dna_command == 'train':
        train_handler(arguments, subparser)

    elif arguments.dna_command == 'evaluate':
        evaluate_handler(arguments, subparser)

    else:
        raise ValueError('A suitable command handler could not be found.')


def main(argv: typing.Sequence):

    parser = argparse.ArgumentParser(prog='dna')

    subparsers = parser.add_subparsers(dest='dna_command', title='commands')
    subparsers.required = True

    split_parser = subparsers.add_parser(
        'split-data', help='creates train and test splits of the data'
    )
    configure_split_parser(split_parser)

    train_parser = subparsers.add_parser(
        'train', help='trains and stores a model'
    )
    configure_train_parser(train_parser)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='loads and evaluates a stored model'
    )
    configure_evaluate_parser(evaluate_parser)

    arguments = parser.parse_args(argv[1:])

    handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
