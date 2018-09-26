import json
from dataset import pipelineDataset
from models import PrimitiveModel

primitive_submodel_dict = {}
standard_input_size = 4 # this will ultimately be the number of metafeatures


# build list of edges for each node (represented as nodes that must be evaluated BEFORE the node in question)
def get_edges_for_node(primitive_names, primitive_name):
    edges = []
    if primitive_names.index(primitive_name) != 0:
        # in this specific case (dealing with completed_pipelines_with_baseline.json), each node points only to the
        # node directly following it in the list of primitive_names
        edges.append(
            primitive_submodel_dict.get(primitive_names.index(primitive_name) - 1, "")
        )
    return edges

# todo: consider storing the submodel pipeline somewhere, could be useful information
# returns a DAG of primitive names represented as a dictionary of:
#       {pName: [list of primitives whose output is needed for input]}
def build_submodel_pipeline(pipeline_data):
    submodel_pipeline = {}
    primitive_names = pipeline_data["job_str"].split("___")

    for primitive_name in primitive_names:
        # get node
        if primitive_submodel_dict[primitive_name] is None:
            primitive_submodel_dict[primitive_name] = PrimitiveModel(standard_input_size)

        # get edges pointing to node
        submodel_pipeline[primitive_name] = get_edges_for_node(primitive_names, primitive_name)

    return submodel_pipeline


def perform_learning(submodel_pipeline, target, metafeatures):
    # todo: make an instance of pipeline regressor
    # todo: run DFS on pipeline graph (submodel_pipeline)
    # how do create a nn.Module subclass execute a dag of modules (rather than a list)?

    return 0


def run_it():
    pipelines = pipelineDataset("completed_pipelines_with_baseline.json")

    for pipeline_data in pipelines:
        submodel_pipeline = build_submodel_pipeline(pipeline_data)
        # todo: compute metafeatures and pass them into perform learning
        perform_learning(submodel_pipeline, target=pipeline_data["test_accuracy"])


if __name__ == "__main__":
    run_it()
