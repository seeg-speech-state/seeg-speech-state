import pandas as pd
import numpy as np
import torch

from enum import Enum
from loguru import logger
from torch.utils.data import Dataset, random_split, Subset
from pathlib import Path
from typing import Literal

from preprocessing import SignalFilter


def collate_fn_time(batch):
    X = [v[0] for v in batch]
    Y = [v[1] for v in batch]
    return torch.cat(X, dim=-1), torch.cat(Y, dim=-1)


class Word300Dataset(Dataset):
    def __init__(self,
                 base_dir: Path, subject: str, drop_index: list[str] = [], channel_config: str | None = None,
                 sig_filter_used: SignalFilter = SignalFilter.DEFAULT, x_dtype = torch.float32, tag: str | None = None):
        self.base_dir = base_dir
        if tag is None:
            tag = subject

        logger.info("loading word300 dataset, base_dir={}, subject={}, tag={}, channel_config={}", base_dir, subject, tag, channel_config)

        data_path = base_dir / f"{tag},data,{sig_filter_used.abbr()}.npz"
        info_path = base_dir / f"{tag},index_info.csv"
        data = np.load(data_path)

        concat_index = np.concat((data['rest_index'], data['speech_index']))
        concat_data = np.concat((data['rest'], data['speech']))

        if channel_config is not None:
            from dataset_info import datasets
            logger.debug("channel_config={}", channel_config)
            sel = datasets[subject]['channel_configs'][channel_config]
            assert isinstance(sel, list)
            # N, C, T
            concat_data = concat_data[:, sel, :]
            self.selected_channels = np.array(sel)
        else:
            self.selected_channels = np.arange(concat_data.shape[1])

        if len(drop_index) > 0:
            where = np.argwhere(np.isin(concat_index, drop_index))
            concat_data = np.delete(concat_data, where, axis=0)
            concat_index = np.delete(concat_index, where, axis=0)

        assert concat_data.shape[0] == concat_index.shape[0]

        x = torch.from_numpy(concat_data).to(x_dtype)
        N, C, T = x.shape

        zeros, ones = data['rest'].shape[0], data['speech'].shape[0]
        y = torch.zeros((N, 2, 1)) # TODO: I'm not sure
        y[:zeros, 0, :] = 1
        y[zeros:, 1, :] = 1
        # y = torch.cat((torch.zeros((zeros, T), dtype=torch.long), torch.ones((ones, T), dtype=torch.long)))

        logger.debug("tag={}, x.shape={}, y.shape={}, zeros={}, ones={}", tag, x.shape, y.shape, zeros, ones)

        assert x.shape[0] == y.shape[0]

        info = pd.read_csv(info_path)

        self.x = x
        self.y = y
        self.index = concat_index
        if len(drop_index) > 0:
            assert not np.isin(self.index, drop_index).any()
        self.info: pd.DataFrame = info


    def random_discard_channels(self, ratio: float, seed: int):
        N, C, T = self.x.shape
        assert len(self.selected_channels) == C

        rng = np.random.default_rng(seed)
        p = rng.permutation(C)[int(C*ratio):]
        self.selected_channels = self.selected_channels[p]
        self.x = self.x[:, p, :]

        logger.debug("got dataset C={}, discard ratio={}, now we have selected_channels={}, C'={}", C, ratio, self.selected_channels, self.x.shape[1])

        return self.selected_channels


    def split(self, way: Literal["sentence", "blender"], ratio: float, seed: int) -> tuple[Subset, Subset]:
        logger.info("split way={}", way)
        if way == "sentence":
            return self._split_by_sentence(ratio, seed)[:2]
        elif way == "blender":
            return self._split_blender(ratio, seed)
        else:
            raise ValueError("Unknown way={} to split dataset", way)


    def _split_blender(self, ratio: float, seed: int) -> tuple[Subset, Subset]:
        gen = torch.Generator().manual_seed(seed)
        s = torch.utils.data.random_split(self, [ratio, 1-ratio], generator=gen)
        assert len(s) == 2
        return tuple(s) # type: ignore


    def _split_by_sentence(self, ratio: float, seed: int) -> tuple[Subset, Subset, pd.Series, pd.Series]:
        shuffled_file_name = self.info.sample(frac=1, random_state=seed).speech_file_name
        mid = int(len(shuffled_file_name) * ratio)
        train_file_name, test_file_name = shuffled_file_name[:mid], shuffled_file_name[mid:]
        assert len(set(train_file_name) & set(test_file_name)) == 0

        train_idx = np.argwhere(np.isin(self.index, train_file_name)).squeeze()
        test_idx = np.argwhere(np.isin(self.index, test_file_name)).squeeze()
        train_dataset, test_dataset = torch.utils.data.Subset(self, train_idx), torch.utils.data.Subset(self, test_idx)

        return train_dataset, test_dataset, train_file_name, test_file_name


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

