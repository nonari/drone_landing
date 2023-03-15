from torch.utils.data import Sampler


class SeqSampler(Sampler[int]):
    def __init__(self, data_source, indices):
        super().__init__(None)
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
