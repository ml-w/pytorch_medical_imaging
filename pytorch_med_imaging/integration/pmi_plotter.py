from numpy._typing import NDArray, ArrayLike
from abc import abstractmethod

class PMIPlotter(object):
    def __init__(self):
        self.num_step: int = 0
        self.num_epoch: int = 0

    @abstractmethod
    def plot_img(self, img: ArrayLike, cmap: str = 'gray') -> None:
        pass

    @abstractmethod
    def log_scalar(self, val: value, label: str) -> None:
        pass

    @abstractmethod
    def save_model_scalar(self, model, tag: str):
        pass

    def plot_weight_histogram(self, *args, **kwargs):
        pass