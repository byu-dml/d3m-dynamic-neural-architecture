from torch import nn


class LanguageModelHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg):
        super(LanguageModelHead, self).__init__()
        self.n_embd = cfg['n_embd']
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(h_trunc)
        return lm_logits
