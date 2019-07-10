import copy

from torch import nn

from .block import Block


class Transformer(nn.Module):

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg['n_embd'])
        self.drop = nn.Dropout(cfg['embd_pdrop'])
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg['n_layer'])])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        # Combine batch size and number of documents
        x = x.view(-1, x.size(-2), x.size(-1))
        embedding = self.embed(x)
        # Add the position information to the input embeddings
        encoding = embedding.sum(dim=2)
        for block in self.h:
            encoding = block(encoding)
        return encoding
