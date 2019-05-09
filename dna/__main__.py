import argparse
import sys
import typing

from data import get_data


def configure_split_parser(parser):
    parser.add_argument(
        '--data-path', type=str, action='store',
        required=True, help='path of train data'
    )


def split_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *, data_resolver=get_data
):
    data = data_resolver(getattr(arguments, 'data_path'))


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
