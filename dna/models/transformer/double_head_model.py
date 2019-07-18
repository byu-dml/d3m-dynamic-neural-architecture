import collections

from torch import nn

from .transformer import Transformer
from .language_model_head import LanguageModelHead
from .task_head import TaskHead


class DoubleHeadModel(nn.Module):
    """ Transformer with language model and task specific heads """
    def __init__(self, cfg, clf_token, num_classes, vocab=40990, sequence_dim=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = Transformer(cfg, vocab=vocab, n_ctx=sequence_dim)
        self.lm_head = LanguageModelHead(self.transformer, cfg)
        self.task_head = TaskHead(clf_token, cfg, num_classes)

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)
        return lm_logits, task_logits
