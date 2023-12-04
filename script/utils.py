import numpy as np
import sys
import os
import csv
import time
import pandas as pd
from abc import abstractmethod, ABC
from typing import Tuple
from copy import deepcopy
from sklearn.model_selection import KFold
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from pytorch_lightning import Trainer
from torch.nn import functional as F


class CVDataModule(ABC):

    def __init__(self,
                 data_module: pl.LightningDataModule,
                 n_splits: int = 10,
                 shuffle: bool = True):
        self.data_module = data_module
        self._n_splits = n_splits
        self._shuffle = shuffle

    @abstractmethod
    def split(self):
        pass


class KFoldCVDataModule(CVDataModule):
    """
        K-fold cross-validation data module

    Args:
        data_module: data module containing data to be split
        n_splits: number of k-fold iterations/data splits
    """

    def __init__(self,
                 data_module: pl.LightningDataModule,
                 n_splits: int = 10):
        super().__init__(data_module, n_splits)
        self._k_fold = KFold(n_splits=self._n_splits, shuffle=self._shuffle)

        # set dataloader kwargs if not available in data module (as in the default one)
        self.dataloader_kwargs = data_module.__getattribute__('dataloader_kwargs') or {}

        #set important defaults if not present
        #self.dataloader_kwargs['batch_size'] = self.dataloader_kwargs.get('batch_size', 32)
        #self.dataloader_kwargs['num_workers'] = self.dataloader_kwargs.get('num_workers', os.cpu_count())
        #self.dataloader_kwargs['shuffle'] = self.dataloader_kwargs.get('shuffle', True)

    def get_data(self):
        """
            Extract and concatenate training and validation datasets from data module.
        """
        self.data_module.setup()
        train_ds = self.data_module.train_dataloader().dataset
        val_ds = self.data_module.val_dataloader().dataset
        return ConcatDataset([train_ds, val_ds])

    def split(self) -> Tuple[DataLoader, DataLoader]:
        """
            Split data into k-folds and yield each pair
        """
        # 0. Get data to split
        data = self.get_data()

        # 1. Iterate through splits
        for train_idx, val_idx in self._k_fold.split(range(len(data))):
            train_dl = DataLoader(Subset(data, train_idx),
                                  **self.dataloader_kwargs)
            val_dl = DataLoader(Subset(data, val_idx),
                                **self.dataloader_kwargs)

            yield train_dl, val_dl


class CVTrainer:

    def __init__(self, trainer: pl.Trainer):
        super().__init__()
        self._trainer = trainer

    @staticmethod
    def _update_logger(logger: LightningLoggerBase, fold_idx: int):
        """
            Change a model logger parameters to log new fold
        Args:
            logger: Logger to update
            fold_idx: Fold ID
        """
        if hasattr(logger, 'experiment_name'):
            logger_key = 'experiment_name'
        elif hasattr(logger, 'name'):
            logger_key = 'name'
        else:
            raise AttributeError('The logger associated with the trainer '
                                 'should have an `experiment_name` or `name` '
                                 'attribute.')
        new_experiment_name = getattr(logger, logger_key) + f'/fold_{fold_idx}'
        setattr(logger, logger_key, new_experiment_name)

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback: ModelCheckpoint, fold_idx: int):
        """
            Update model checkpoint object with fold information
        Args:
            model_ckpt_callback: Model checkpoint object
            fold_idx: Fold ID
        """
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def update_loggers(self, trainer: Trainer, fold_idx: int):
        """
            Change model's loggers parameters to log new fold
        Args:
            trainer: Trainer whose logger to update
            fold_idx: Fold ID
        """
        if not isinstance(trainer.logger, LoggerCollection):
            _loggers = [trainer.logger]
        else:
            _loggers = trainer.logger

        # Update loggers:
        for _logger in _loggers:
            self._update_logger(_logger, fold_idx)

    def fit(self, model: pl.LightningModule, data: CVDataModule):
        for fold_idx, loaders in enumerate(data.split()):

            # Clone model & trainer:
            _model = deepcopy(model)
            _trainer = deepcopy(self._trainer)

            # Update loggers and callbacks:
            #self.update_loggers(_trainer, fold_idx)
            for callback in _trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # fit
            _trainer.fit(_model, *loaders)


class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,padding_value=0)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor([x[1] for x in sorted_batch])

        return sequences_padded, lengths, labels
    
class PadSequence_inf:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,padding_value=0)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.DoubleTensor([x[1] for x in sorted_batch])

        return sequences_padded, lengths, labels

class PadImage:
    def __call__(self, batch):
        images = [x[0] for x in batch]
        labels = torch.LongTensor([x[1] for x in batch])
        adr_id = torch.LongTensor([x[2] for x in batch])

        max_height = max([img.size(1) for img in images])
        max_width = max([img.size(2) for img in images])

        images_padded = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]).tolist() for img in images]
        images_padded = torch.DoubleTensor(images_padded)

        return images_padded, labels, adr_id
    
class PadImage_inf:
    def __call__(self, batch):
        images = [x[0] for x in batch]
        bars = torch.DoubleTensor([x[1] for x in batch])
        #bars = torch.LongTensor([x[1] for x in batch])
        adr_id = torch.LongTensor([x[2] for x in batch])
        label = torch.LongTensor([x[3] for x in batch])
        
        max_height = max([img.size(1) for img in images])
        max_width = max([img.size(2) for img in images])

        images_padded = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]).tolist() for img in images]
        images_padded = torch.DoubleTensor(images_padded)

        return images_padded, bars, adr_id, label

class PadImage_inf_comp:
    def __call__(self, batch):
        images = [x[0] for x in batch]
        images_comp = [x[1] for x in batch]
        output_label = torch.LongTensor([x[2] for x in batch])
        #bars = torch.DoubleTensor([x[1] for x in batch])
        #adr_id = torch.LongTensor([x[2] for x in batch])
        #label = torch.LongTensor([x[3] for x in batch])
        
        max_height = max([img.size(1) for img in images])
        max_width = max([img.size(2) for img in images])

        images_padded = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]).tolist() for img in images]
        images_padded = torch.DoubleTensor(images_padded)
        
        images_padded_comp = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]).tolist() for img in images_comp]
        images_padded_comp = torch.DoubleTensor(images_padded_comp)

        return images_padded, images_padded_comp, output_label
    
class PadImage_inf_comp_val:
    def __call__(self, batch):
        images = [x[0] for x in batch]
        bars = torch.DoubleTensor([x[1] for x in batch])
        adr_id = torch.LongTensor([x[2] for x in batch])
        label = torch.LongTensor([x[3] for x in batch])
        
        max_height = max([img.size(1) for img in images])
        max_width = max([img.size(2) for img in images])

        images_padded = [F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)]).tolist() for img in images]
        images_padded = torch.DoubleTensor(images_padded)

        return images_padded, bars, adr_id, label
    
class PadImage_pack:
    def __call__(self, batch):
        images = [x[0] for x in batch]
        labels = torch.LongTensor([x[1] for x in batch])

        images_padded = [[img[:,:,i:i+50].tolist() for i in [0,50,100,150,200,250]] for img in images]
        images_padded = torch.DoubleTensor(images_padded)
        
        images_padded = images_padded.view((images_padded.shape[0],6,128,50))

        return images_padded, labels


from typing import Callable

# from ImbalancedDatasetSampler library
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        y_labels = []
        for xx in dataset:
            y_labels.append(xx[1])

        labels = torch.LongTensor(y_labels)
        return labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
