from torch import nn


class TaskHead(nn.Module):
    """Classification Head for the transformer"""

    def __init__(self, clf_token, cfg, n_output):
        super(TaskHead, self).__init__()
        self.n_embd = cfg['n_embd']
        self.clf_token = clf_token
        self.dropout = nn.Dropout2d(cfg['clf_pdrop'])  # To reproduce the noise_shape parameter of TF implementation
        self.n_output = n_output
        self.linear = nn.Linear(cfg['n_embd'], n_output)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = clf_h.view(-1, x.size(1), self.n_embd)
        # This double transposition is there to replicate the behavior
        # of the noise_shape argument in the tensorflow
        # implementation.  For more details, see
        # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = clf_h.contiguous().view(-1, self.n_embd)
        output = self.linear(clf_h)

        output = output.view(-1, x.size(1)*self.n_output)
        return output
