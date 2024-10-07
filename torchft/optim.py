from typing import TYPE_CHECKING, Optional

from torch.optim import Optimizer

if TYPE_CHECKING:
    from torchft.manager import Manager


class OptimizerWrapper(Optimizer):
    def __init__(self, manager: "Manager", optim: Optimizer) -> None:
        self.optim = optim
        self.manager = manager

    def add_param_group(self, param_group: object) -> None:
        self.optim.add_param_group(param_group)

    def load_state_dict(self, state_dict: object) -> None:
        self.optim.load_state_dict(state_dict)

    def state_dict(self) -> object:
        return self.optim.state_dict()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.manager.step()
        self.optim.zero_grad(set_to_none)

    def step(self, closure: Optional[object] = None) -> None:
        assert closure is None, "optimizers that use closures are not supported"
        if self.manager.should_commit():
            self.optim.step()
