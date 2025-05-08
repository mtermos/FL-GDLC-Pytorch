# tests/test_data_module.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import pytest

from src.data.data_module import ClientTrainDataModule, ServerEvalDataModule


@pytest.fixture
def small_data():
    # 10 samples, 3 features each
    x_train = np.arange(30, dtype=float).reshape(10, 3)
    y_train = np.arange(10, dtype=int) % 2   # binary labels
    x_val = np.arange(20, dtype=float).reshape(10, 2)
    y_val = np.arange(10, dtype=int) % 3     # three‐class labels
    return x_train, y_train, x_val, y_val


def test_setup_creates_datasets(small_data):
    x_train, y_train, x_val, y_val = small_data
    dm = ClientTrainDataModule(x_train, y_train, x_val, y_val, batch_size=4)
    # Before setup these attributes don't exist
    assert not hasattr(dm, "train_dataset")
    dm.setup()
    # After setup they do, and lengths match
    assert isinstance(dm.train_dataset, TensorDataset)
    assert isinstance(dm.val_dataset,   TensorDataset)
    assert len(dm.train_dataset) == len(x_train)
    assert len(dm.val_dataset) == len(x_val)
    # Tensors have correct dtype
    tx, ty = dm.train_dataset.tensors
    assert tx.dtype == torch.float32
    assert ty.dtype == torch.int64


def test_train_dataloader_properties_and_batches(small_data):
    x_train, y_train, x_val, y_val = small_data
    dm = ClientTrainDataModule(x_train, y_train, x_val, y_val, batch_size=4)
    dm.setup()
    loader = dm.train_dataloader()
    # Should be a DataLoader
    assert isinstance(loader, DataLoader)
    # Batch size and workers
    assert loader.batch_size == 4
    assert loader.num_workers == 0
    # Shuffle => RandomSampler
    assert isinstance(loader.sampler, RandomSampler)
    # Collect all batches and verify shapes
    all_x, all_y = [], []
    for batch_x, batch_y in loader:
        all_x.append(batch_x)
        all_y.append(batch_y)
        # each batch has shape (<=4, feature_dim)
        assert batch_x.ndim == 2
        assert batch_y.ndim == 1
    # concatenate should recover whole dataset
    x_cat = torch.cat(all_x, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    # In total we saw exactly len(x_train) samples
    assert x_cat.shape[0] == len(x_train)
    assert y_cat.shape[0] == len(y_train)
    # And contents match (as a multiset, since shuffle=True)
    assert set(y_cat.tolist()) == set(y_train.tolist())


@pytest.mark.parametrize("do_validate, expect_loader", [(True, True), (False, False)])
def test_val_dataloader_toggle(small_data, do_validate, expect_loader):
    x_train, y_train, x_val, y_val = small_data
    dm = ClientTrainDataModule(x_train, y_train, x_val, y_val,
                               batch_size=5, do_validate=do_validate)
    dm.setup()
    vd = dm.val_dataloader()
    if expect_loader:
        assert isinstance(vd, DataLoader)
        # Since batch_size=5 and len(x_val)==10, we get exactly 2 batches
        assert len(list(vd)) == 2
    else:
        # Should be an empty list
        assert vd == []


def test_test_dataloader_and_batches():
    # ServerEvalDataModule uses only test split
    x_test = np.arange(12, dtype=float).reshape(6, 2)
    y_test = np.arange(6, dtype=int)
    tm = ServerEvalDataModule(x_test, y_test, batch_size=4)
    tm.setup()
    loader = tm.test_dataloader()
    assert isinstance(loader, DataLoader)
    # Two batches: sizes 4 and 2
    batches = list(loader)
    assert len(batches) == 2
    bx0, by0 = batches[0]
    bx1, by1 = batches[1]
    assert bx0.shape == (4, 2)
    assert by0.shape == (4,)
    assert bx1.shape == (2, 2)
    assert by1.shape == (2,)
    # Check data matches
    all_y = torch.cat([by0, by1], dim=0).tolist()
    assert all_y == y_test.tolist()
