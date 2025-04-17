import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class FLDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        # Convert numpy arrays to PyTorch tensors
        self.x_train = torch.FloatTensor(x_train)
        # self.y_train = torch.FloatTensor(y_train) if len(
        #     y_train.shape) == 1 else torch.LongTensor(y_train)
        self.y_train = torch.LongTensor(y_train)

        self.x_val = torch.FloatTensor(x_val)
        self.y_val = torch.LongTensor(y_val)

    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.val_dataset = TensorDataset(self.x_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)


class TestDataModule(pl.LightningDataModule):
    def __init__(self, x_test, y_test, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.x_test = torch.FloatTensor(x_test)
        self.y_test = torch.LongTensor(y_test)

    def setup(self, stage=None):
        self.test_dataset = TensorDataset(self.x_test, self.y_test)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)
