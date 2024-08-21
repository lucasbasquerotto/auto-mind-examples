import os
import typing
import torch
import seaborn as sns
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from auto_mind.supervised.handlers import (
    MetricsCalculator, MetricsCalculatorParams, BatchExecutor, BatchExecutorParams)
from auto_mind.supervised.data import DatasetGroup
from auto_mind.supervised.data import MinimalFullState
import warnings

M = typing.TypeVar("M")

COLOR_TRAIN = '#1f77b4'
COLOR_VALIDATION = '#ff7f0e'
COLOR_TEST = '#2ca02c'

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def get_color(key: str):
    if key == 'train':
        return COLOR_TRAIN
    elif key == 'validation':
        return COLOR_VALIDATION
    elif key == 'test':
        return COLOR_TEST
    else:
        return None

MAX_BARS = 3

class MetricsPlotter(MetricsCalculator):
    def plot(self, info: MinimalFullState, metrics: dict[str, typing.Any], figsize: tuple[float, float] | None) -> list[Figure]:
        raise NotImplementedError

CategoricItem = tuple[typing.Any, torch.Tensor | int]
C = typing.TypeVar('C', bound=CategoricItem)

BinaryMultiLabelItem = tuple[typing.Any, torch.Tensor | list[int]]
B = typing.TypeVar('B', bound=BinaryMultiLabelItem)

class MetricsFileDirectPlotter(MetricsCalculator):
    def __init__(self, plotter: MetricsPlotter, file_path: str, figsize: tuple[float, float] | None = None):
        self.plotter = plotter
        self.file_path = file_path
        self.figsize = figsize

    def run(self, params: MetricsCalculatorParams) -> dict[str, typing.Any]:
        plotter = self.plotter
        file_path = self.file_path
        info = params.info
        figsize = self.figsize

        result = plotter.run(params)
        figs = plotter.plot(info=info, metrics=result, figsize=figsize)

        base_path = os.path.dirname(file_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        pdf = PdfPages(file_path)
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

        return dict()

class MetricsItemPlotter(typing.Generic[M]):
    def __init__(self, name: str):
        self.name = name

    def run(self, params: MetricsCalculatorParams) -> M:
        raise NotImplementedError

    def plot(self, info: MinimalFullState, metric: M, figsize: tuple[float, float] | None) -> list[Figure]:
        raise NotImplementedError

    def as_plotter(self) -> MetricsPlotter:
        return MetricsListPlotter(items=[self])

    def as_file_plotter(self, file_path: str, figsize: tuple[float, float] | None = None) -> MetricsFileDirectPlotter:
        return MetricsFileDirectPlotter(
            plotter=self.as_plotter(),
            file_path=file_path,
            figsize=figsize)

T = typing.TypeVar('T', bound=MetricsItemPlotter)

class MetricsListPlotter(MetricsPlotter, typing.Generic[T]):
    def __init__(self, items: list[T | None]):
        self.items = [i for i in items if i is not None]

    def run(self, params: MetricsCalculatorParams) -> dict[str, typing.Any]:
        result: dict[str, typing.Any] = { item.name: item.run(params) for item in self.items }
        return result

    def plot(self, info: MinimalFullState, metrics: dict[str, typing.Any], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []
        for item in self.items:
            metric = metrics.get(item.name)
            if metric is not None:
                figs += item.plot(info=info, metric=metric, figsize=figsize)
        return figs

class MainMetrics(MetricsItemPlotter[dict[str, None]]):
    def __init__(self, name: str, no_loss=False, no_accuracy=False, no_time=False):
        super().__init__(name=name)
        self.no_loss = no_loss
        self.no_accuracy = no_accuracy
        self.no_time = no_time

    def run(self, params: MetricsCalculatorParams) -> dict[str, None]:
        return dict()

    def plot(self, info: MinimalFullState, metric: dict[str, None], figsize: tuple[float, float] | None) -> list[Figure]:
        train_losses: list[tuple[int, float]] = info.train_results.losses
        train_accuracies: list[tuple[int, float]] = info.train_results.accuracies
        train_times: list[tuple[int, float]] = info.train_results.times

        val_losses: list[tuple[int, float]] | None = info.train_results.val_losses
        val_accuracies: list[tuple[int, float]] | None = info.train_results.val_accuracies
        val_times: list[tuple[int, float]] | None = info.train_results.val_times

        fig: Figure | None = None

        if train_losses and train_accuracies:
            nrows = 0
            nrows += 1 if not self.no_loss else 0
            nrows += 1 if not self.no_accuracy else 0
            nrows += 1 if not self.no_time else 0
            fig, ax_untyped = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, squeeze=False)
            fig.tight_layout()
            ax = typing.cast(np.ndarray, ax_untyped)
            next_ax = 0

            if not self.no_loss:
                ax1: Axes = ax[next_ax, 0]
                next_ax += 1
                epochs = [epoch for epoch, _ in train_losses]
                losses = [loss for _, loss in train_losses]
                ax1.plot(epochs, losses, '-', label='Train Loss', color=COLOR_TRAIN)
                if val_losses and len(val_losses) == len(epochs):
                    epochs = [epoch for epoch, _ in val_losses]
                    losses = [loss for _, loss in val_losses]
                    ax1.plot(epochs, losses, '-', label='Validation Loss', color=COLOR_VALIDATION)
                ax1.set_ylabel('Loss')

            if not self.no_accuracy:
                ax2: Axes = ax[next_ax, 0]
                next_ax += 1
                epochs = [epoch for epoch, _ in train_accuracies]
                accuracies = [accuracy for _, accuracy in train_accuracies]
                ax2.plot(epochs, accuracies, '-', label='Train Accuracy', color=COLOR_TRAIN)
                if val_accuracies and len(val_accuracies) == len(epochs):
                    epochs = [epoch for epoch, _ in val_accuracies]
                    accuracies = [accuracy for _, accuracy in val_accuracies]
                    ax2.plot(epochs, accuracies, '-', label='Validation Accuracy', color=COLOR_VALIDATION)
                ax2.set_ylabel('Accuracy')

            if not self.no_time:
                ax3: Axes = ax[next_ax, 0]
                next_ax += 1
                epochs = [epoch for epoch, _ in train_times]
                times = [time for _, time in train_times]
                ax3.plot(epochs, times, '-', label='Train Time', color=COLOR_TRAIN)
                if val_times and len(val_times) == len(epochs):
                    epochs = [epoch for epoch, _ in val_times]
                    times = [time for _, time in val_times]
                    ax3.plot(epochs, times, '-', label='Validation Time', color=COLOR_VALIDATION)
                ax3.set_ylabel('Time/Epoch (ms)')

            test_accuracy = info.test_results.accuracy if info.test_results else None
            info_accuracy = f' - Accuracy: {(test_accuracy*100):.2f}%' if test_accuracy else ''

            ax[0, 0].set_title(f'[{self.name}] Main Metrics{info_accuracy}')
            ax[-1, 0].set_xlabel('Epochs')

            for ax_item in ax:
                ax_typed: Axes = ax_item[0]
                ax_typed.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return [fig] if fig else []

class DatasetsAmountsMetrics(MetricsItemPlotter[dict[str, int]]):
    def __init__(self, name: str, datasets: DatasetGroup[typing.Any]):
        super().__init__(name=name)
        self.datasets = datasets

    def dataset_amount(self, dataset: Dataset[typing.Any] | None) -> int:
        if not dataset:
            return 0

        if isinstance(dataset, typing.Sized):
            return len(dataset)
        else:
            return len(list([d for d in dataset]))

    def run(self, params: MetricsCalculatorParams) -> dict[str, int]:
        return dict(
            train=self.dataset_amount(self.datasets.train),
            validation=self.dataset_amount(self.datasets.validation),
            test=self.dataset_amount(self.datasets.test),
        )

    def plot(self, info: MinimalFullState, metric: dict[str, int], figsize: tuple[float, float] | None) -> list[Figure]:
        keys: list[str] = list(metric.keys())
        values: list[int] = list(metric.values())
        fig = plt.figure(figsize=figsize)

        if len(keys) <= MAX_BARS:
            plt.bar(keys, values, color=PALETTE)
        else:
            plt.barh(keys, values, color=PALETTE)

        plt.title(f"[{self.name}] Datasets - Amounts")
        return [fig]

class DatasetsCategoricLabelsMetrics(MetricsItemPlotter[dict[str, list[int]]], typing.Generic[C]):
    def __init__(self, name: str, categories: list[str], datasets: DatasetGroup[C]):
        super().__init__(name=name)
        self.datasets = datasets
        self.categories = categories

    def dataset_amounts(self, dataset: Dataset[C] | None) -> list[int]:
        result = [0] * len(self.categories)

        if not dataset:
            return result

        for input, target in dataset:
            label_idx = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
            diff = (label_idx+1) - len(result)

            if diff > 0:
                result += [0] * diff

            result[label_idx] += 1

        return result

    def run(self, params: MetricsCalculatorParams) -> dict[str, list[int]]:
        return dict(
            train=self.dataset_amounts(self.datasets.train),
            validation=self.dataset_amounts(self.datasets.validation),
            test=self.dataset_amounts(self.datasets.test),
        )

    def plot(self, info: MinimalFullState, metric: dict[str, list[int]], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []

        for key, all_values in metric.items():
            names: list[str] = []
            values: list[int] = []

            for label_idx, value in enumerate(all_values):
                if value:
                    names.append(self.categories[label_idx])
                    values.append(value)

            fig = plt.figure(figsize=figsize)

            if len(names) <= MAX_BARS:
                plt.bar(names, values, color=PALETTE)
            else:
                plt.barh(names, values, color=PALETTE)

            plt.title(f"[{self.name}] Datasets - Categoric Labels - {key}")
            figs.append(fig)

        return figs

class DatasetsBinaryLabelsMetrics(MetricsItemPlotter[dict[str, list[tuple[int, int]]]], typing.Generic[B]):
    def __init__(self, name: str, labels: list[str], datasets: DatasetGroup[B]):
        super().__init__(name=name)
        self.datasets = datasets
        self.labels = labels

    def dataset_amounts(self, dataset: Dataset[B] | None) -> list[tuple[int, int]]:
        result = [(0, 0)] * len(self.labels)

        if not dataset:
            return result

        for input, target in dataset:
            items = target.detach().tolist() if isinstance(target, torch.Tensor) else target
            for i, t in enumerate(items):
                if i < len(result):
                    value = float(t) >= 0.5
                    result[i] = (result[i][0] + (not value), result[i][1] + value)

        return result

    def run(self, params: MetricsCalculatorParams) -> dict[str, list[tuple[int, int]]]:
        return dict(
            train=self.dataset_amounts(self.datasets.train),
            validation=self.dataset_amounts(self.datasets.validation),
            test=self.dataset_amounts(self.datasets.test),
        )

    def plot(self, info: MinimalFullState, metric: dict[str, list[tuple[int, int]]], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []

        for key, all_values in metric.items():
            data: list[tuple[str, str, int]] = []

            for label_idx, (v_false, v_true) in enumerate(all_values):
                if (v_false or v_true) and (label_idx < len(self.labels)):
                    label = self.labels[label_idx]
                    data.append(('False', label, v_false))
                    data.append(('True', label, v_true))

            df = pd.DataFrame(data, columns=['group', 'label', 'val'])
            fig = plt.figure(figsize=figsize)
            sns.barplot(data=df, x='val', y='label', hue='group', orient='v' if len(df) <= MAX_BARS else 'h')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(f"[{self.name}] Datasets - Binary Labels - {key}")
            figs.append(fig)

        return figs

class TruePredictedClassMetrics(MetricsItemPlotter[dict[str, typing.Any]], typing.Generic[C]):
    def __init__(self, name: str, categories: list[str], dataset: Dataset[C], executor: BatchExecutor[typing.Any, typing.Any] | None = None):
        super().__init__(name=name)
        self.categories = categories
        self.dataset = dataset
        self.executor = executor

    def confusion_matrix(self, model: torch.nn.Module) -> tuple[np.ndarray, int]:
        with torch.no_grad():
            dataset = self.dataset
            executor = self.executor

            model.eval()

            size = len(self.categories)
            matrix = np.zeros((size, size), dtype=np.int32)
            counter = 0

            for input, target in dataset:
                input_batch = input.unsqueeze(0) if isinstance(input, torch.Tensor) else [input]

                if executor:
                    params = BatchExecutorParams(
                        model=model,
                        input=input_batch,
                        last_output=None)
                    full_output = executor.run(params)
                    output = executor.main_output(full_output)
                else:
                    output: torch.Tensor = model(input_batch)

                idx_out = int(output.squeeze(0).argmax().item())
                idx_target = int(target.item()) if isinstance(target, torch.Tensor) else int(target)

                if idx_target >= 0 and idx_target < size and idx_out >= 0 and idx_out < size:
                    counter += 1
                    matrix[idx_target, idx_out] += 1

            return matrix, counter

    def run(self, params: MetricsCalculatorParams) -> dict[str, typing.Any]:
        confmat, amount = self.confusion_matrix(params.model)
        return dict(confmat=confmat, amount=amount)

    def plot(self, info: MinimalFullState, metric: dict[str, typing.Any], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []
        category_names = self.categories
        amount = metric.get('amount', 0)
        confmat: np.ndarray | None = metric.get('confmat')

        if confmat is not None:
            fig = plt.figure(figsize=figsize)
            sns.heatmap(confmat, annot=True, fmt='', cmap='Blues', xticklabels=category_names, yticklabels=category_names)
            plt.title(f"[{self.name}] True vs Predicted Classes - Amount: {amount}")
            plt.ylabel('True')
            plt.xlabel('Predicted')
            figs.append(fig)

        return figs

class TruePredictedBinaryMetrics(MetricsItemPlotter[dict[str, typing.Any]], typing.Generic[B]):
    def __init__(self, name: str, labels: list[str], dataset: Dataset[B]):
        super().__init__(name=name)
        self.labels = labels
        self.categories = ['False', 'True']
        self.dataset = dataset

    def confusion_matrix(self, model: torch.nn.Module) -> np.ndarray:
        with torch.no_grad():
            dataset = self.dataset

            model.eval()

            n_labels = len(self.labels)
            size = 2
            matrix = np.zeros((n_labels, size, size), dtype=np.int32)

            for input, target in dataset:
                output: torch.Tensor = model(input.unsqueeze(0))[0]

                for i in range(n_labels):
                    idx_out = round(output[i].item())
                    idx_target = int(target[i].item()) if isinstance(target, torch.Tensor) else int(target[i])

                    if idx_target >= 0 and idx_target < size and idx_out >= 0 and idx_out < size:
                        matrix[i, idx_target, idx_out] += 1

            return matrix

    def run(self, params: MetricsCalculatorParams) -> dict[str, np.ndarray]:
        confmats = self.confusion_matrix(params.model)
        result: dict[str, np.ndarray] = dict()

        for i, label in enumerate(self.labels):
            confmat = confmats[i]
            result[label] = confmat

        return result

    def plot(self, info: MinimalFullState, metric: dict[str, np.ndarray], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []
        category_names = self.categories

        for label in self.labels:
            confmat: np.ndarray | None = metric.get(label)

            if confmat is not None:
                fig = plt.figure(figsize=figsize)
                sns.heatmap(confmat, annot=True, fmt='', cmap='Blues', xticklabels=category_names, yticklabels=category_names)
                plt.title(f"[{self.name}] [{label}] True vs Predicted Classes")
                plt.ylabel('True')
                plt.xlabel('Predicted')
                figs.append(fig)

        return figs

class SHAPMetrics(MetricsItemPlotter[dict[str, typing.Any]]):
    def __init__(self, name: str, features: list[str], datasets: DatasetGroup[tuple[torch.Tensor, torch.Tensor]]):
        super().__init__(name=name)
        self.datasets = datasets
        self.features = features

    def _get_shap(self):
        with warnings.catch_warnings():
            # filter shap/plots/colors/_colorconv.py:819:
            # Converting `np.inexact` or `np.floating` to a dtype is deprecated
            warnings.simplefilter("ignore", category=DeprecationWarning)
            import shap

        return shap

    def run(self, params: MetricsCalculatorParams) -> dict[str, typing.Any]:
        model = params.model
        datasets = self.datasets.limit(100)

        train_dataset = datasets.train
        test_dataset = datasets.test if datasets.test else datasets.validation

        if not train_dataset or not test_dataset:
            return dict()

        train_data = torch.stack([input for input, _ in train_dataset])
        test_data = torch.stack([input for input, _ in test_dataset])

        shap = self._get_shap()
        e = shap.DeepExplainer(model, train_data)
        shap_values = e.shap_values(test_data, check_additivity=False)

        return dict(shap_values=shap_values, data=test_data, expected=e.expected_value)

    def plot(self, info: MinimalFullState, metric: dict[str, typing.Any], figsize: tuple[float, float] | None) -> list[Figure]:
        figs: list[Figure] = []
        shap_values: np.ndarray | None = metric.get('shap_values')
        data: np.ndarray | None = metric.get('data')

        if shap_values is not None and data is not None:
            shap = self._get_shap()
            shap.summary_plot(
                shap_values,
                data,
                feature_names=np.array(self.features),
                max_display=len(self.features),
                show=False,
                figsize=figsize)
            fig = plt.gcf()
            figs.append(fig)

        return figs
