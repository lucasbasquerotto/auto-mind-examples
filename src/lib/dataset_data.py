import typing
from typing import Generic, TypeVar
import torch
from auto_mind.supervised.data import DatasetGroup

I = TypeVar("I")
T = TypeVar("T")
L = TypeVar("L")

class Datasource(Generic[I]):
    def __init__(self, datasets: DatasetGroup[I]):
        self.datasets = datasets

class LabeledDatasource(Datasource[tuple[I, T]], Generic[I, T, L]):
    def __init__(
            self,
            datasets: DatasetGroup[tuple[I, T]],
            all_labels: list[L]):

        super().__init__(datasets=datasets)

        self.all_labels = all_labels

class LabeledTensorDatasource(
    LabeledDatasource[torch.Tensor, torch.Tensor, typing.Any]
):
    pass
