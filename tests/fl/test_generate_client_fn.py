import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

# import the thing under test
from src.fl.fl_client import FLClient
from src.fl.generate_client_fn import generate_client_fn


class DummyTBLogger:
    def __init__(self, save_dir):
        # record the path it was given
        self.save_dir = save_dir


class DummyWBLogger:
    def __init__(self, project, config, version, name, save_dir):
        self.project = project
        self.config = config
        self.version = version
        self.name = name
        self.save_dir = save_dir


@pytest.fixture(autouse=True)
def patch_external(monkeypatch):
    # 1) Patch init_model so we capture its inputs and return a dummy model
    captured = {}

    def dummy_init_model(training_cfg, model_cfg, input_dim, labels_mapping, weight_tensor, using_wandb):
        captured['args'] = {
            'training_cfg': training_cfg,
            'model_cfg': model_cfg,
            'input_dim': input_dim,
            'labels_mapping': labels_mapping,
            'weight_tensor': weight_tensor,
            'using_wandb': using_wandb,
        }
        # return a dummy "model" object
        return SimpleNamespace(_weight_tensor=weight_tensor)
    monkeypatch.setattr(
        "src.fl.generate_client_fn.init_model", dummy_init_model)

    # 2) Patch FLClient.to_client to return self so we can inspect it
    monkeypatch.setattr(FLClient, "to_client", lambda self: self)

    # 3) Patch the two loggers
    monkeypatch.setattr(
        "src.fl.generate_client_fn.TensorBoardLogger", DummyTBLogger)
    monkeypatch.setattr("src.fl.generate_client_fn.WandbLogger", DummyWBLogger)

    return captured


class LoggingWrapper:
    """Minimal wrapper so cfg_base.logging[...] and .selected_type both work"""

    def __init__(self, selected_type, tb_cfg, wb_cfg):
        self.selected_type = selected_type
        self._cfgs = {"tensorboard": tb_cfg, "wandb": wb_cfg}

    def __getitem__(self, key):
        return self._cfgs[key]


@pytest.fixture
def cfg_base(tmp_path):
    # experiment name
    exp = SimpleNamespace(name="my_exp")
    # logging sub-configs
    tb_cfg = SimpleNamespace(save_dir=str(tmp_path/"tb"))
    wb_cfg = SimpleNamespace(project="proj", save_dir=str(tmp_path/"wb"))
    logging = LoggingWrapper(selected_type=None, tb_cfg=tb_cfg, wb_cfg=wb_cfg)

    # assemble base config
    return SimpleNamespace(
        experiment=exp,
        logging=logging,
        dataset_properties=SimpleNamespace(val_size=0.4),
        random_seed=123,
        training=SimpleNamespace(
            max_epochs=1, batch_size=8, use_weighted_loss=True, weighted_loss_version="v4"),
        fl=SimpleNamespace(clients_to_val=[0])
    )


@pytest.fixture
def model_cfg():
    return SimpleNamespace(model=SimpleNamespace(name="mymodel"))


@pytest.fixture
def data_and_labels():
    # two clients, each with 10 samples of dim-2
    data = [np.arange(20).reshape(10, 2), np.arange(30, 50).reshape(10, 2)]
    # labels as pandas Series so .value_counts() works
    labels = [
        pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    ]
    return data, labels


def make_context(client_id):
    # Flower Context only matters for .node_config
    return SimpleNamespace(node_config={"partition-id": str(client_id)})


def test_tensorboard_branch(cfg_base, model_cfg, data_and_labels, patch_external):
    data, labels = data_and_labels
    cfg_base.logging.selected_type = "tensorboard"

    client_fn = generate_client_fn(
        data=data,
        labels=labels,
        model_cfg=model_cfg,
        cfg_base=cfg_base,
        exp_type="EXP",
        config_to_add_to_logger={"foo": "bar"},
        run_dtime="20250101-000000",
        input_dim=2,
        labels_mapping={0: "A", 1: "B"},
    )

    # pick client_id = 1 --> in [0,5]? No, so do_validate=False, tensorboard branch
    ctx = make_context(1)
    client = client_fn(ctx)

    # verify we got back the FLClient instance
    assert isinstance(client, FLClient)

    # data_module was constructed with the right batch_size & do_validate
    dm = client.data_module
    assert dm.batch_size == cfg_base.training.batch_size
    assert dm.do_validate is False

    # train/test split: 10 examples, 40% val => 6 train, 4 val
    dm.setup()
    assert len(dm.train_dataset) == 6
    assert len(dm.val_dataset) == 4

    # logger should be our DummyTBLogger
    assert isinstance(client.logger, DummyTBLogger)
    # and its save_dir should contain experiment name & exp_type & client_id
    assert "EXP_mymodel" in client.logger.save_dir
    # assert "_client_1" in client.logger.save_dir

    # verify init_model was called with correct args
    args = patch_external['args']
    # input_dim and labels_mapping should match
    assert args['input_dim'] == 2
    assert args['labels_mapping'] == {0: "A", 1: "B"}
    # using_wandb=False in this branch
    assert args['using_wandb'] is False

    # check weight_tensor logic:
    wt = args['weight_tensor'].cpu().numpy()
    # there are 5 zeros and 5 ones in labels[1]
    # counts_arr = [5,5], total=10, so weight per class = 10/(2*5)=1.0
    assert np.allclose(wt, np.array([1.0, 1.0]))


def test_wandb_branch(cfg_base, model_cfg, data_and_labels, patch_external):
    data, labels = data_and_labels
    cfg_base.logging.selected_type = "wandb"

    client_fn = generate_client_fn(
        data=data,
        labels=labels,
        model_cfg=model_cfg,
        cfg_base=cfg_base,
        exp_type="EXP",
        config_to_add_to_logger={"foo": "bar"},
        run_dtime="20250101-000000",
        input_dim=2,
        labels_mapping={0: "A", 1: "B"},
    )

    # pick client_id = 0 --> in [0,5], so do_validate=True, wandb branch
    ctx = make_context(0)
    client = client_fn(ctx)

    assert isinstance(client, FLClient)
    dm = client.data_module
    assert dm.do_validate is True

    # logger should be our DummyWBLogger
    lg = client.logger
    assert isinstance(lg, DummyWBLogger)
    # project and config should be passed through
    assert lg.project == cfg_base.logging['wandb'].project
    assert lg.config == {"foo": "bar"}
    assert lg.version.startswith("20250101-000000_mymodel_0")
    assert f"client_0" in lg.name
    assert lg.save_dir.endswith("my_exp/EXP_mymodel_client_0")

    # init_model using_wandb=True
    assert patch_external['args']['using_wandb'] is True

    # the weight tensor for labels[0] also should be ones
    wt = patch_external['args']['weight_tensor'].cpu().numpy()
    assert np.allclose(wt, np.array([1.0, 1.0]))
