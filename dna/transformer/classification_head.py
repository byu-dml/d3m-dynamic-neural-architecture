from torch import nn


class ClassificationHead(nn.Module):
    """Classification Head for the transformer

    TODO: test this class."""
    def __init__(self, clf_token, cfg, n_class):
        super(ClassificationHead, self).__init__()
        self.n_embd = cfg['n_embd']
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg['clf_pdrop'])
        self.linear = nn.Linear(cfg['n_embd'], n_class)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits