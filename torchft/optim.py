from torch.optim import Optimizer


class FTOptimizer(Optimizer):
    def __init__(self, optim: Optimizer) -> None:
        self.optim = optim

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optim.zero_grad(set_to_none)
