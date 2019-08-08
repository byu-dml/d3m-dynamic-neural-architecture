from .attention_regression_model import AttentionRegressionModel
from .torch_modules.explicit_dag_attention_mlp import ExplicitDAGAttentionMLP
from dna.data import group_json_objects


class ExplicitDAGAttentionRegressionModel(AttentionRegressionModel):

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience, *,
        output_dir=None, verbose=False
    ):
        self._get_pipeline_structures(train_data)

        super().fit(
            train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
        )

    @staticmethod
    def _modify_pipeline_structure(structure):
        for i, inputs in enumerate(structure):
            inputs = sorted(inputs)
            input_paths = []
            for input_ in inputs:
                input_path = []
                if input_ != 'inputs.0':
                    input_path.append(input_)
                input_path.append(i)
                input_paths.append(input_path)

            structure[i] = input_paths

    def _get_model(self, train_data):
        n_features = len(train_data[0][self.features_key])
        return ExplicitDAGAttentionMLP(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            in_features=self.num_primitives,
            attention_in_features=self.attention_in_features,
            attention_activation_name=self.attention_activation_name,
            attention_hidden_features=self.attention_hidden_features,
            dropout=self.dropout,
            reduction_name=self.reduction_name,
            use_mask=self.use_mask,
            mlp_extra_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            mlp_activation_name=self.activation_name,
            output_size=1,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            device=self.device,
            seed=self._model_seed
        )
