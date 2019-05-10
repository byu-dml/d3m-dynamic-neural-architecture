import random
from typing import List, Dict

from data import group_json_objects
import torch


class SiameseDataLoader(object):
    """
    Rather than iterating through all the batches in the data set per epoch, it iterates through a constant amount
    of batches strictly less than the total number of batches. Some of these batches may be repeats. A batch is
    created by randomly pairing two pipelines and dynamically creating metafeature and target tensors based on
    the performance of those pipelines on the data sets that both of them share. This save memory by dynamically
    creating the batches and throwing them away when they're done. It is because even just one epoch would be too large
    that we only provide a subset of the total possible batches in an epoch.
    """

    def __init__(
        self, data: List[Dict], group_key: str, pipeline_key: str,
            data_set_key: str, features_key: str, target_key: str, device: str
    ):
        self.data = data
        self.group_key = group_key
        self.pipeline_key = pipeline_key
        self.data_set_key = data_set_key
        self.features_key = features_key
        self.target_key = target_key
        self.device = device
        self.data_set_info_key = 'data_set_info'

        self._init_group_data_sets()
        self.num_batches_per_epoch = len(self.group_data_sets)
        self.pipeline_ids = list(self.group_data_sets.keys())

    def _init_group_data_sets(self):
        # Group the data
        grouped_data = group_json_objects(self.data, self.group_key)

        # Create a data set, technically a data set of "data sets", for each pipeline
        self.group_data_sets = {}
        for group, group_indices in grouped_data.items():
            group_data = [self.data[i] for i in group_indices]

            # Since all the pipelines are the same in this group, being grouped by pipeline, just get the first one
            pipeline = group_data[0][self.pipeline_key]

            # Order the metafeatures and f1 values, corresponding to this pipeline, by data set
            # This allows the pipelines to be paired and compared later
            group_data = self.prepare_group_data_for_pairing(group_data, pipeline)

            self.group_data_sets[group] = group_data

    def prepare_group_data_for_pairing(self, group_data, pipeline, ):
        # Create a dictionary containing the data set information corresponding to the pipeline of this group
        # The data set information includes the metafeatures of each data set and how well the pipeline performed on it
        data_set_info = {}

        for item in group_data:
            data_set = item[self.data_set_key]
            metafeatures = item[self.features_key]
            f1 = item[self.target_key]

            data_set_info[data_set] = {self.features_key: metafeatures, self.target_key: f1}

        # Return a dictionary containing the pipeline and its corresponding data set information
        return {self.pipeline_key: pipeline,
                self.data_set_info_key: data_set_info}

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        for i in range(self.num_batches_per_epoch):
            groups_identical = True
            group1 = random.choice(self.pipeline_ids)
            while groups_identical:
                group2 = random.choice(self.pipeline_ids)
                groups_identical = group1 == group2

            group1_data = self.group_data_sets[group1]
            group2_data = self.group_data_sets[group2]

            group1_data_set_info = group1_data[self.data_set_info_key]
            group2_data_set_info = group2_data[self.data_set_info_key]
            data_set_ids = []
            x_batch = []
            y_batch = []

            # Create a batch consisting of the intersect between the two groups
            for data_set_id in group1_data_set_info:
                # If the data set is found in both groups
                if data_set_id in group2_data_set_info:
                    data_set_ids.append(data_set_id)
                    group1_info = group1_data_set_info[data_set_id]
                    group2_info = group2_data_set_info[data_set_id]
                    metafeatures = group1_info[self.features_key]

                    # Since the data sets should be the same, so should their metafeatures
                    assert metafeatures == group2_info[self.features_key]

                    # Create the target based on comparing the performance of each pipeline on this data set
                    group1_f1 = group1_info[self.target_key]
                    group2_f1 = group2_info[self.target_key]
                    if group1_f1 > group2_f1:
                        target = 0
                    else:
                        target = 1

                    x_batch.append(metafeatures)
                    y_batch.append(target)

            # Convert the metafeatures and target of this data set with this pipeline to tensors
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            y_batch = torch.tensor(y_batch, dtype=torch.int64, device=self.device)

            # Get the pipelines of each group
            pipeline1 = group1_data[self.pipeline_key]
            pipeline2 = group2_data[self.pipeline_key]

            yield ((group1, group2), (pipeline1, pipeline2), x_batch, data_set_ids), y_batch
        raise StopIteration()

    def __len__(self):
        return self.num_batches_per_epoch
