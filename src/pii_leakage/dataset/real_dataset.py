# The code is provided by the paper "Analyzing Leakage of Personally Identifiable Information in Language Models"
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from copy import deepcopy

from datasets import load_dataset

from ..arguments.dataset_args import DatasetArgs
from ..arguments.env_args import EnvArgs
from ..arguments.ner_args import NERArgs
from .dataset import Dataset
from ..global_configs import system_configs


class RealDataset(Dataset):
    """ A lightweight wrapper around a huggingface text dataset
    that allows caching and indexing PII
    """

    def __init__(self, dataset_args: DatasetArgs, ner_args: NERArgs = None, env_args: EnvArgs = None):
        """ A real dataset is a dataset loaded from real training data.
        """
        self.dataset_args = dataset_args
        super().__init__(ner_args, env_args)
        self._tokenized_base_dataset = self._load_base_dataset()  # this is the reference to the huggingface dataset

    def _load_base_dataset(self, force_undefended=False):
        """ Loads the huggingface dataset. """
        load_dataset_params = {
            "path": self.dataset_args.dataset_path,
            "cache_dir": self.dataset_args.cache_dir(),
            "name": self.dataset_args.dataset_mode if not force_undefended else "undefended",
        }

        if self.dataset_args.sample_duplication_rate != 1:
            load_dataset_params["sample_duplication_rate"] = self.dataset_args.sample_duplication_rate

        return load_dataset(**load_dataset_params)[self.dataset_args.split]


    @property
    def _pii_cache(self):
        """ Returns the filepath for the file that contains all pii and their location. """
        return os.path.join(os.path.abspath(system_configs.CACHE_DIR), self.dataset_args.hash(suffix="pii"))

    def shuffle(self):
        self._base_dataset.shuffle()
        return self

    def copy(self):
        return deepcopy(self)

    def select(self, indices):
        clone = self.copy()
        clone._tokenized_base_dataset = clone._tokenized_base_dataset.select(indices)
        return clone

    def __iter__(self):
        return self._base_dataset.__iter__()

    def __getitem__(self, item):
        return self._base_dataset.__getitem__(item)

    def get_hf_dataset(self):
        return self._base_dataset

    def first(self, column_name="text"):
        return self[column_name][0]
