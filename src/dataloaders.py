import torch
from pathlib import Path
from hydra.utils import to_absolute_path

from src.datasets import PurchasesDataset


class PurchasesLoader:
    def __init__(self, data_dir="../data/",
                 x_filename="x_train.csv", y_filename="y_train.csv",
                 batch_size=8, mode="train",
                 drop_last=False, shuffle=True, num_workers=8):
        data_dir = Path(to_absolute_path(data_dir))
        x_path = data_dir / x_filename
        y_path = data_dir / y_filename
        dataset = PurchasesDataset(x_path, y_path)
        self.loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            drop_last=drop_last
        )