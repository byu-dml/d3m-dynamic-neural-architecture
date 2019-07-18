class DNASiameseModule(nn.Module):

    def __init__(self, input_model, submodules, output_model):
        super().__init__()
        self.input_model = input_model
        self.submodules = submodules
        self.output_model = output_model
        self.h1 = None
        self.f_activation = F_ACTIVATIONS[ACTIVATION]

    def forward(self, args):
        pipeline_ids, (left_pipeline, right_pipeline), x = args
        self.h1 = self.input_model(x)
        left_h2 = self.recursive_get_output(left_pipeline, len(left_pipeline) - 1)
        right_h2 = self.recursive_get_output(right_pipeline, len(right_pipeline) - 1)
        h2 = torch.cat((left_h2, right_h2), 1)
        return self.output_model(h2)

    def recursive_get_output(self, pipeline, current_index):
        """
        The recursive call to find the input
        :param pipeline: the pipeline list containing the submodules
        :param current_index: the index of the current submodule
        :return:
        """
        try:
            current_submodule = self.submodules[pipeline[current_index]['name']]
            if "inputs.0" in pipeline[current_index]['inputs']:
                return self.f_activation(current_submodule(self.h1))

            outputs = []
            for input in pipeline[current_index]["inputs"]:
                curr_output = self.recursive_get_output(pipeline, input)
                outputs.append(curr_output)

            if len(outputs) > 1:
                new_output = self.f_activation(current_submodule(torch.cat(tuple(outputs), dim=1)))
            else:
                new_output = self.f_activation(current_submodule(curr_output))

            return new_output
        except Exception as e:
            print("There was an error in the foward pass.  It was ", e)
            print(pipeline[current_index])
            quit(1)


