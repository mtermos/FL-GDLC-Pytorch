import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pytorch_lightning as pl


class ClientTrainDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, batch_size=32, do_validate=True, oversample=True):
        super().__init__()
        self.batch_size = batch_size
        self.do_validate = do_validate

        # Convert numpy arrays to PyTorch tensors
        self.x_train = torch.FloatTensor(x_train)
        # self.y_train = torch.FloatTensor(y_train) if len(
        #     y_train.shape) == 1 else torch.LongTensor(y_train)
        self.y_train = torch.LongTensor(y_train)

        self.x_val = torch.FloatTensor(x_val)
        self.y_val = torch.LongTensor(y_val)
        self.oversample = oversample

    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.val_dataset = TensorDataset(self.x_val, self.y_val)

    def train_dataloader(self):
        if self.oversample:
            # compute sample weights based on class frequency
            y_np = self.y_train.numpy()
            class_counts = np.bincount(y_np)
            # class_weights = 1.0 / class_counts
            inv_freq = 1.0 / class_counts

            alpha = 0.5
            class_weights = inv_freq ** alpha

            sample_weights = class_weights[y_np]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                # num_samples=int(len(sample_weights) * 2),
                replacement=True,
            )
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        if not self.do_validate:
            return []
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)


class ServerEvalDataModule(pl.LightningDataModule):
    def __init__(self, x_test, y_test, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.x_test = torch.FloatTensor(x_test)
        self.y_test = torch.LongTensor(y_test)

    def setup(self, stage=None):
        self.test_dataset = TensorDataset(self.x_test, self.y_test)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)
