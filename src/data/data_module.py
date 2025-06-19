import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
import pytorch_lightning as pl


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, target_idx=None):
        """
        Dataset for time series data that creates sequences on-the-fly.

        Args:
            data: Input data tensor of shape (n_samples, n_features)
            sequence_length: Length of sequences to create
            target_idx: Optional tensor of target indices. If None, uses the last position in sequence
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Calculate start index for the sequence
        start_idx = max(0, idx - self.sequence_length + 1)

        # Get the sequence
        sequence = self.data[start_idx:idx + 1]

        # If sequence is shorter than sequence_length, pad with zeros at the beginning
        if len(sequence) < self.sequence_length:
            padding = torch.zeros(self.sequence_length -
                                  len(sequence), sequence.shape[1])
            sequence = torch.cat([padding, sequence], dim=0)

        # Get target index (either from target_idx or use the last position)
        target_idx = self.target_idx[idx] if self.target_idx is not None else -1

        return sequence, target_idx


class ClientTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=32,
        do_validate=True,
        oversample=True,
        use_sequences=False,
        sequence_length=None
    ):
        """
        Initialize the client training data module.

        Args:
            x_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels
            batch_size: Batch size for training
            do_validate: Whether to perform validation
            oversample: Whether to oversample minority classes
            use_sequences: Whether to use sequence-based data loading (for LSTM/GRU)
            sequence_length: Length of sequences (required if use_sequences=True)
        """
        super().__init__()
        self.batch_size = batch_size
        self.do_validate = do_validate
        self.use_sequences = use_sequences
        self.sequence_length = sequence_length

        if use_sequences and sequence_length is None:
            raise ValueError(
                "sequence_length must be provided when use_sequences=True")

        # Convert numpy arrays to PyTorch tensors
        self.x_train = torch.FloatTensor(x_train)
        self.y_train = torch.LongTensor(y_train)
        self.x_val = torch.FloatTensor(x_val)
        self.y_val = torch.LongTensor(y_val)
        self.oversample = oversample

    def setup(self, stage=None):
        if self.use_sequences:
            # Create sequence-based datasets
            self.train_dataset = TimeSeriesDataset(
                self.x_train,
                sequence_length=self.sequence_length,
                target_idx=self.y_train
            )
            self.val_dataset = TimeSeriesDataset(
                self.x_val,
                sequence_length=self.sequence_length,
                target_idx=self.y_val
            )
        else:
            # Create regular datasets for single-record models
            self.train_dataset = TensorDataset(self.x_train, self.y_train)
            self.val_dataset = TensorDataset(self.x_val, self.y_val)

    def train_dataloader(self):
        if self.oversample:
            # compute sample weights based on class frequency
            y_np = self.y_train.numpy()
            class_counts = np.bincount(y_np)
            inv_freq = 1.0 / class_counts

            alpha = 0.5
            class_weights = inv_freq ** alpha

            sample_weights = class_weights[y_np]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
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
    def __init__(
        self,
        x_test,
        y_test,
        batch_size=32,
        use_sequences=False,
        sequence_length=None
    ):
        """
        Initialize the server evaluation data module.

        Args:
            x_test: Test features
            y_test: Test labels
            batch_size: Batch size for evaluation
            use_sequences: Whether to use sequence-based data loading (for LSTM/GRU)
            sequence_length: Length of sequences (required if use_sequences=True)
        """
        super().__init__()
        self.batch_size = batch_size
        self.use_sequences = use_sequences
        self.sequence_length = sequence_length

        if use_sequences and sequence_length is None:
            raise ValueError(
                "sequence_length must be provided when use_sequences=True")

        self.x_test = torch.FloatTensor(x_test)
        self.y_test = torch.LongTensor(y_test)

    def setup(self, stage=None):
        if self.use_sequences:
            self.test_dataset = TimeSeriesDataset(
                self.x_test,
                sequence_length=self.sequence_length,
                target_idx=self.y_test
            )
        else:
            self.test_dataset = TensorDataset(self.x_test, self.y_test)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)
