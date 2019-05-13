import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

from d3m import index as d3m_index
from d3m import runtime as runtime_module
from d3m.metadata import base as metadata_base, problem
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

import autosklearn.classification


def get_data_loading_pipeline():

    pipeline_description = Pipeline(context=metadata_base.Context.TESTING)
    pipeline_description.add_input(name='inputs')

    # 0 Denormalize Dataset
    denormalize_primitive: PrimitiveBase = d3m_index.get_primitive(
        'd3m.primitives.data_transformation.denormalize.Common'
    )
    step_0 = PrimitiveStep(primitive=denormalize_primitive)
    step_0.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='inputs.0'
    )
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # 1 Dataset to DataFrame
    dataset_to_dataframe_primitive = d3m_index.get_primitive(
        'd3m.primitives.data_transformation.dataset_to_dataframe.Common'
    )
    step_1 = PrimitiveStep(primitive=dataset_to_dataframe_primitive)
    step_1.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='steps.0.produce'
    )
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # 2 Column Parser
    column_parser_primitive = d3m_index.get_primitive(
        'd3m.primitives.data_transformation.column_parser.DataFrameCommon'
    )
    step_2 = PrimitiveStep(primitive=column_parser_primitive)
    step_2.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='steps.1.produce'
    )
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    # 3 Label Encoder
    label_encoder_primitive = d3m_index.get_primitive(
        'd3m.primitives.data_preprocessing.label_encoder.DataFrameCommon'
    )
    step_3 = PrimitiveStep(primitive=label_encoder_primitive)
    step_3.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='steps.2.produce'
    )
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)

    pipeline_description.add_output(name='output', data_reference='steps.3.produce')

    return pipeline_description


class D3MDatasetUtil:

    def __init__(
        self, dataset_name, dataset_dir='/datasets/training_datasets/LL0'
    ):
        self.dataset_name = dataset_name
        self.dataset_path = self.dataset_path = os.path.join(
            dataset_dir, dataset_name
        )
        self.dataset_doc_path = os.path.join(
            self.dataset_path, dataset_name + '_dataset', 'datasetDoc.json'
        )
        self.data_splits_path = os.path.join(
            self.dataset_path, dataset_name + '_problem', 'dataSplits.csv'
        )
        self.problem_path = os.path.join(
            self.dataset_path, dataset_name + '_problem', 'problemDoc.json'
        )

        self.problem_description = problem.parse_problem_description(self.problem_path)
        self.problem_type = self.problem_description['problem']['task_type']
        self.target_col = self.problem_description['inputs'][0]['targets'][0]['column_name']

    def load_data(self, pipeline: Pipeline = None):
        dataset = runtime_module.get_dataset(self.dataset_doc_path)

        if pipeline is None:
            pipeline = get_data_loading_pipeline()

        runtime = runtime_module.Runtime(
            pipeline, context=metadata_base.Context.TESTING
        )

        result = runtime.fit([dataset], return_values=['outputs.0'])
        result.check_success()

        dataset = result.values['outputs.0']
        dataset.drop('d3mIndex', axis=1, inplace=True)

        self.dataset = dataset

    def get_splits(self, *, split_xy=True):
        data_splits = pd.read_csv(self.data_splits_path)
        train_split_indices = (data_splits['fold'] == 0) & (data_splits['type'] == 'TRAIN')
        train_data = self.dataset[train_split_indices]
        test_data = self.dataset[~train_split_indices]

        # for fold in split_df.fold.unique():  # cross validation folds

        if split_xy:
            
            y_train_data = train_data[self.target_col]
            X_train_data = train_data.drop(self.target_col, axis=1)
            y_test_data = test_data[self.target_col]
            X_test_data = test_data.drop(self.target_col, axis=1)
            return X_train_data, y_train_data, X_test_data, y_test_data

        else:
            return train_data, test_data


def compute_auto_sklearn_baseline(dataset_name):
    dataset_util = D3MDatasetUtil(dataset_name)
    dataset_util.load_data()
    X_train, y_train, X_test, y_test = dataset_util.get_splits()

    if dataset_util.problem_type == 'CLASSIFICATION':
        classifier = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=600,
            per_run_time_limit=30,
        )
        classifier.fit(X_train, y_train)
        y_train_predict = classifier.predict(X_train)
        y_test_predict = classifier.predict(X_test)

    # TODO: dynamically get score type
    print(
        f1_score(y_train, y_train_predict, average='macro'),
        f1_score(y_test, y_test_predict, average='macro')
    )
#     Traceback (most recent call last):
#   File "auto_sklearn_baslines.py", line 157, in <module>
#     compute_auto_sklearn_baseline(dataset_name)
#   File "auto_sklearn_baslines.py", line 154, in main
    
#   File "auto_sklearn_baslines.py", line 149, in compute_auto_sklearn_baseline
#     print(
#   File "/usr/local/lib/python3.6/dist-packages/sklearn/metrics/classification.py", line 714, in f1_score
#     sample_weight=sample_weight)
#   File "/usr/local/lib/python3.6/dist-packages/sklearn/metrics/classification.py", line 828, in fbeta_score
#     sample_weight=sample_weight)
#   File "/usr/local/lib/python3.6/dist-packages/sklearn/metrics/classification.py", line 1036, in precision_recall_fscore_support
#     (pos_label, present_labels))
# ValueError: pos_label=1 is not a valid label: array(['10583991021311951928', '16349079934775498145'], dtype='<U20')

def main():
    dataset_name = sys.argv[1]
    compute_auto_sklearn_baseline(dataset_name)

if __name__ == '__main__':
    main()