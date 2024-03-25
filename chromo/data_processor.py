import h5py
import numpy as np
import os
from filelock import FileLock

from chromo import chrombpnet_utils


class DataProcessor:
    def __init__(self, dataset="deepstarr", subsample=False):
        self.dataset = dataset
        self.subsample = subsample

    def load_data(self, data_split, **kwargs):
        if self.dataset == "deepstarr":
            return self.load_deepstarr_data(
                data_split=data_split, subsample=self.subsample
            )
        elif self.dataset == "plantstarr":
            return self.load_plantstarr_data(
                data_split=data_split, subsample=self.subsample
            )
        elif self.dataset == "scbasset":
            return self.load_scbasset_data(
                data_split=data_split, subsample=self.subsample
            )
        elif self.dataset == "hepg2":
            return self.load_hepg2_data(data_split=data_split, subsample=self.subsample)
        elif self.dataset == "chrombpnet":
            return self.load_chrombpnet_data(
                data_split=data_split, subsample=self.subsample, **kwargs
            )
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")

    def load_plantstarr_data(
        self,
        data_split: str,
        data_dir="/home/chandana/projects/deepstarr/data/plantstarr_data.h5",
        subsample: bool = False,
        dataset: str = "leaf",
    ) -> (np.ndarray, np.ndarray):
        """Load dataset"""

        # load sequences and labels
        if data_split == "valid":
            return None, None

        with h5py.File(data_dir, "r") as f:
            x = f[dataset][f"x_{data_split}"][:]
            y = f[dataset][f"y_{data_split}"][:]
        if subsample:
            x = x[: int(x.shape[0] * 0.2)]
            y = y[: int(y.shape[0] * 0.2)]
        return x, y

    def load_scbasset_data(
        self,
        data_split: str,
        data_dir="/home/chandana/projects/hominid_pipeline/data/pbmc.h5",
        subsample: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """Load dataset"""

        # load sequences and labels
        with FileLock(os.path.expanduser(f"{data_dir}.lock")):
            with h5py.File(data_dir, "r") as dataset:
                x = np.array(dataset[f"X_{data_split}"]).astype(np.float32)
                y = np.array(dataset[f"Y_{data_split}"]).astype(np.float32)
        if subsample:
            if data_split == "train":
                x = x[:80000]
                y = y[:80000]
            elif data_split == "valid":
                x = x[:20000]
                y = y[:20000]
            else:
                x = x[:10000]
                y = y[:10000]
        return x, y

    def load_deepstarr_data(
        self,
        data_split: str,
        data_dir="/home/chandana/projects/hominid_pipeline/data/deepstarr_data.h5",
        subsample: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """Load dataset"""

        # load sequences and labels
        with FileLock(os.path.expanduser(f"{data_dir}.lock")):
            with h5py.File(data_dir, "r") as dataset:
                x = np.array(dataset[f"x_{data_split}"]).astype(np.float32)
                y = np.array(dataset[f"y_{data_split}"]).astype(np.float32).transpose()
        if subsample:
            if data_split == "train":
                x = x[:80000]
                y = y[:80000]
            elif data_split == "valid":
                x = x[:20000]
                y = y[:20000]
            else:
                x = x[:10000]
                y = y[:10000]
        return x, y

    def load_hepg2_data(
        self,
        data_split: str,
        data_dir="/home/chandana/projects/hominin/data/HepG2_onehot_static.h5",
        subsample: bool = False,
    ) -> (np.ndarray, np.ndarray):
        """Load dataset"""

        # load sequences and labels
        with FileLock(os.path.expanduser(f"{data_dir}.lock")):
            with h5py.File(data_dir, "r") as dataset:
                x = np.array(dataset[f"x_{data_split}"]).astype(np.float32)
                y = np.array(dataset[f"y_{data_split}"]).astype(np.float32)
                print(y.shape)
        if subsample:
            if data_split == "train":
                x = x[:20000]
                y = y[:20000]
            elif data_split == "valid":
                x = x[:10000]
                y = y[:10000]
            else:
                x = x[:50000]
                y = y[:50000]
        return x, y

    def load_chrombpnet_data(
        self,
        data_split: str,
        data_dir="",
        subsample: bool = False,
        return_coords=False,
        **kwargs
    ):
        """Load dataset"""

        data_gen = chrombpnet_utils.initialize_generators(
            mode=data_split, smoke_test=subsample, return_coords=return_coords, **kwargs
        )
        return data_gen

    @staticmethod
    def shape_info(x_data, y_data):
        N, L, A = x_data.shape
        output_shape = y_data.shape[-1]
        print(f"Input shape: {N, L, A}. Output shape: {output_shape}")
        return (L, A), output_shape
