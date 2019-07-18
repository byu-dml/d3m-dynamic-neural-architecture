from torch import nn


class SimilarityHead(nn.Module):
    """ Similarity Head for the transformer

        TODO: test this class."""
    def __init__(self, clf_token, cfg):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg['n_embd']
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg['clf_pdrop'])
        self.linear = nn.Linear(cfg['n_embd'], 1)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        sim_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        sim_h = sim_h[flat == self.clf_token, :]
        sim_h = self.dropout(sim_h)
        sim_h = sim_h.sum(dim = 1)
        sim_logits = self.linear(sim_h)

        return sim_logits