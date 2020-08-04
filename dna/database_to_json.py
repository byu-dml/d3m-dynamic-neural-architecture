import json
import re
from dateutil.parser import parse
import collections
import os

from bson import json_util
import pymongo
from tqdm import tqdm

from dna.utils import has_path


try:
    real_mongo_port = int(os.environ['REAL_MONGO_PORT'])
    lab_hostname = os.environ['LAB_HOSTNAME']
except Exception as E:
    print("ERROR: environment variables not set")
    raise E


def flatten(d, parent_key='', sep='_'):
    """
    This flattens a dictionary
    :param d: the dictionary to be flattened
    :param parent_key: the token used to indicate it came from a previous key
    :param sep: the seperator between parent and child
    :return: a flattened non-string dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    items = dict(items)
    # remove info like PCA primitive ID
    items_not_strings = {k: v for k, v in items.items() if type(v) != str}
    return dict(items_not_strings)


class DatabaseToJson:

    supported_problem_types = {
        # `d3m_metric_name` is the name of the metric as it exists in
        # the D3M pipeline run documents. `dna_metric_name` is the name we use for
        # the metric in this repo.
        "classification": {"d3m_metric_name": "F1_MACRO", "dna_metric_name": "test_f1_macro"},
        "regression": {"d3m_metric_name": "ROOT_MEAN_SQUARED_ERROR", "dna_metric_name": "test_rmse"}
    }

    def __init__(self):
        self.connect_to_mongo()

    def connect_to_mongo(self, host_name=lab_hostname, mongo_port=real_mongo_port):
        """
        Connects and returns a session to the mongo database
        :param host_name: the host computer that has the database server
        :param mongo_port: the port number of the database
        :return: a MongoDB session
        """
        try:
            self.mongo_client = pymongo.MongoClient(host_name, mongo_port)
        except Exception as e:
            print("Cannot connect to the Mongo Client at port {}. Error is {}".format(mongo_port, e))

    def get_time_elapsed(self, pipeline_run):
        """
        A helper function for finding the time of the pipeline_run
        :param pipeline_run: a dictionary-like object containing the pipeline run
        :return: the total time in seconds it took the pipeline to execute
        """
        begin = pipeline_run["start"]
        begin_val = parse(begin)
        end = pipeline_run["end"]
        end_val = parse(end)
        total_time = (end_val - begin_val).total_seconds()
        return total_time

    def get_pipeline_from_run(self, pipeline_run):
        """
        This function gets the pipeline that corresponds to a pipeline run
        :param pipeline_run: the produce pipeline run
        :return: the pipeline that corresponds to the pipeline_run
        """
        db = self.mongo_client.metalearning
        collection = db.pipelines
        pipeline_doc = collection.find({"$and": [{"id": pipeline_run["pipeline"]["id"]},
                                                 {"digest": pipeline_run["pipeline"]["digest"]}]})[0]
        return pipeline_doc

    def get_pipeline_run_info(self, pipeline_run):
        """
        Collects and gathers the data needed for the DNA system from a pipeline run
        :param pipeline_run: the pipeline run object to be summarized
        :return: a dictionary object summarizing the pipeline run. `None` is returned
            if the pipeline run is not useable for the metadataset. 
        """
        pipeline = pipeline_run["pipeline"]
        pipeline_run_id = pipeline_run["id"]
        simple_pipeline  = self.parse_simpler_pipeline(pipeline)
        if simple_pipeline is None:
            # This pipeline is not useable for the metadataset.
            return None
        problem_type = self.get_problem_type(pipeline_run["problem"])
        if problem_type == "unsupported":
            return None
        test_score = self.get_score(
            pipeline_run,
            self.supported_problem_types[problem_type]["d3m_metric_name"]
        )
        if test_score is None:
            # This pipeline run was not scored using the supported metric for this
            # problem type.
            return None
        dataset_id = pipeline_run["datasets"][0]["id"]
        test_predict_time = self.get_time_elapsed(pipeline_run)
        # TODO have this find the fit pipeline and get the train score
        # and train predict time, then include those in the returned dictionary.
        pipeline_run_info = {
            "pipeline": simple_pipeline,
            "pipeline_run_id": pipeline_run_id,
            "problem_type": problem_type,
            "dataset_id": dataset_id,
            self.supported_problem_types[problem_type]["dna_metric_name"]: test_score,
            "test_time": test_predict_time,
        }
        return pipeline_run_info
    
    def get_score(self, pipeline_run: dict, score_name: str):
        """
        Finds and returns the score identified by `score_name`. If there
        is no score by that name, returns `None`.
        """
        for score_dict in pipeline_run["run"]["results"]["scores"]:
            if score_dict["metric"]["metric"] == score_name:
                return score_dict["value"]
        # There was no score found under `score_name`.
        return None

    def get_metafeature_info(self, pipeline_run):
        """
        Collects and gathers the data needed for the DNA system from a metafeature
        :param dataset_name: the name/id of the dataset
        :return: a dictionary object summarizing the dataset in metafeatures
        """
        db = self.mongo_client.metalearning
        collection = db.metafeatures
        try:
            mf_pipeline_run = collection.find_one({"$and": [{"datasets.id": pipeline_run["datasets"][0]["id"]},
                                                     {"datasets.digest": pipeline_run["datasets"][0]["digest"]}]})
            mf_steps = [step for step in mf_pipeline_run["steps"] if has_path(step, ["method_calls", 1, "metadata", "produce", 0, "metadata", "data_metafeatures"])]
            if len(mf_steps) != 1:
                raise AssertionError(f"expected metafeatures pipeline run to have 1 metafeatures step, not {len(mf_steps)}")
            features = mf_steps[0]["method_calls"][1]["metadata"]["produce"][0]["metadata"]["data_metafeatures"]
            features_flat = flatten(features)
            # TODO: implement this
            metafeatures_time = 0
            return {"metafeatures": features_flat, "metafeatures_time": metafeatures_time}
        except Exception as e:
            # don't use this pipeline_run
            return {}

    def collect_pipeline_runs(self) -> None:
        """
        This is the main function that collects, and writes to file, all pipeline runs and metafeature information
        It writes the file to data/complete_pipelines_and_metafeatures.json
        :param mongo_client: a connection to the Mongo database
        """
        db = self.mongo_client.analytics
        collection = db.pipeline_runs
        # For our metadataset, we only need pipeline runs
        # that have scores associated with them; PRODUCE phase runs that
        # completed successfully.
        pipeline_run_filter = {"run.phase": "PRODUCE", "status.state": "SUCCESS"}
        n_runs = collection.count(pipeline_run_filter)
        pipeline_run_cursor = collection.find(pipeline_run_filter)

        print("Collecting and distilling pipeline runs into a metadataset...")
        list_of_experiments = {problem_type: [] for problem_type in self.supported_problem_types}
        for index, pipeline_run in tqdm(enumerate(pipeline_run_cursor), total=n_runs):
            pipeline_run_info = self.get_pipeline_run_info(pipeline_run)
            if pipeline_run_info is None:
                # This pipeline run is not useable for the metadataset.
                continue
            metafeatures = self.get_metafeature_info(pipeline_run)
            # TODO: get all metafeatures so we don't need this
            if metafeatures != {}:
                experiment_json = dict(pipeline_run_info, **metafeatures)
                list_of_experiments[experiment_json["problem_type"]].append(experiment_json)

        for problem_type in list_of_experiments.keys():
            final_data_file = json.dumps(list_of_experiments[problem_type], sort_keys=True, indent=4, default=json_util.default)
            final_data_path = "data/complete_pipelines_and_metafeatures_test_{}.json".format(problem_type)
            with open(final_data_path, "w") as file:
                file.write(final_data_file)
            print(f"Wrote metadataset for problem type {problem_type} to path {final_data_path}.")


    def is_phrase_in(self, phrase, text):
        """
        A simple regex search
        :param phrase: the phrase to search for
        :param text: the text to be searched
        :return:
        """
        return re.search(r"\b{}\b".format(phrase), text, re.IGNORECASE) is not None

    def get_problem_type(self, problem):
        """
        This function finds the problem type from a problem description.
        :param problem: a d3m problem description
        :return: a string containing the type of problem
        """
        # First, find the keywords characterizing the problem.
        if has_path(problem, ["problem", "task_keywords"]):
            task_keywords = {w.lower() for w in problem["problem"]["task_keywords"]}
        elif has_path(problem, ["problem", "task_type"]):
            task_keywords = {
                problem["problem"]["task_type"].lower(),
                problem["problem"]["task_subtype"].lower()
            }
        else:
            raise ValueError(
                "Could not find any keywords identifying "
                f"the task type for problem:\n{json.dumps(problem, indent=4)}"
            )
        
        # Next determine the problem type(s) from those keywords.
        problem_types = {keyword for keyword in task_keywords if keyword in self.supported_problem_types}
        if len(problem_types) == 1:
            return problem_types.pop()
        elif len(problem_types) > 1:
            raise AssertionError(
                "Problem has more than one supported problem "
                f"type {problem_types}. Don't know which to use."
            )
        # No supported problem types were found in the problem description.
        return "unsupported"

    def parse_simpler_pipeline(self, full_pipeline):
        """
        This function takes a pipeline object from D3M and turns it into a list of dictionaries where
        each dictionary is a primitive containing the primitive name and the inputs (a list of ints)

        :param full_pipeline: the full d3m pipeline
        :return: The simplified pipeline. `None` is returned if the pipeline is not useable for the
            metadataset.
        """
        pipeline_steps = full_pipeline["steps"]
        simplified_steps = []
        for pipeline_step in pipeline_steps:
            pipeline_step_name = pipeline_step["primitive"]["python_path"]
            inputs_list = []
            if "arguments" not in pipeline_step:
                # This pipeline is not using the normal arguments API (it may be a keras pipeline),
                # so we don't have an accurate view of what its computation graph actually is,
                # so we won't use it for this metadataset.
                return None
            for key, value in pipeline_step["arguments"].items():
                inputs = value["data"]
                pipeline_step_inputs = self.parse_inputs(inputs)
                inputs_list += pipeline_step_inputs
                
            # add info to our pipeline
            simplified_steps.append({"name": pipeline_step_name, "inputs": inputs_list})

        return { "id": full_pipeline["id"], "steps": simplified_steps }
    
    def parse_inputs(self, inputs) -> list:
        """
        Handles the case where D3M primitive step input strings can be either
        a list or a string. 
        """
        if isinstance(inputs, str):
            return [self.parse_input_string(inputs)]
        elif isinstance(inputs, list):
            return [self.parse_input_string(input) for input in inputs]
        else:
            raise ValueError(f"unsupported inputs type {type(inputs)}")

    def parse_input_string(self, string_name: str):
        """
        This helper function parses the input name from the D3M version (aka `steps.0.produce` to 0)
        :param string_name: the string name from D3M
        :return: the simplified name of the input
        """
        list_of_parts = string_name.split(".")
        if list_of_parts[0] == "inputs":
            return string_name
        else:
            # return only the integer part
            return int(list_of_parts[1])


if __name__ == "__main__":
    db_to_json = DatabaseToJson()
    db_to_json.collect_pipeline_runs()