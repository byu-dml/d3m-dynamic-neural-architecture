import argparse

from dna.__main__ import configure_evaluate_parser, configure_split_parser, split_handler


def get_evaluate_args(model: str, model_config_path: str, data_path_train):
    parser = argparse.ArgumentParser()

    configure_evaluate_parser(parser)
    argv = [
        '--model', model,
        '--model-config-path', model_config_path,
        '--model-seed', '0',
        '--problem', 'regression', 'rank', 'subset',
        '--k', '2',
        '--train-path', data_path_train,
        '--test-size', '2',
        '--split-seed', '0',
        '--metafeature-subset', 'all',
        '--no-cache',
    ]
    arguments = parser.parse_args(argv)
    return arguments


def split_data(data_path_train: str, raw_data_path: str):
    # Make sure the data is unzipped and ready
    parser = argparse.ArgumentParser()
    configure_split_parser(parser)
    argv = [
        "--data-path", raw_data_path,
        "--train-path", data_path_train,
        "--test-path", 'data/small_classification_test.json',
        "--test-size", "2",
        '--split-seed', "0"
    ]
    arguments = parser.parse_args(argv)
    split_handler(arguments)
