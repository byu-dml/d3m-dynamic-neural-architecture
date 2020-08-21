

class NMN(nn.Module):
    """
    A Neural Module Network
    """

    def __init__(self, *, seed, device):
        super().__init__()

        self.device = device
        self.seed = seed

        self._dag_lstm = DAGLSTM()
        self._dna = DNA()
        self._mlp = Submodule()

    def forward(self, args):
        pass
