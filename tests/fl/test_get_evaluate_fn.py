import time
import numpy as np
import pytest
from types import SimpleNamespace

# Adjust this import to wherever you defined get_evaluate_fn
from src.fl.get_evaluate_fn import get_evaluate_fn


class LoggingWrapper:
    """So cfg_base.logging[selected_type] works."""

    def __init__(self, selected_type, tb_cfg, wb_cfg):
        self.selected_type = selected_type
        self._cfgs = {"tensorboard": tb_cfg, "wandb": wb_cfg}

    def __getitem__(self, key):
        return self._cfgs[key]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    captured = {}

    # 1) Fix time.strftime for deterministic TensorBoardLogger paths
    monkeypatch.setattr(time, "strftime", lambda fmt: "FIXEDTIME")

    # 2) Dummy loggers
    class DummyTBLogger:
        def __init__(self, save_dir):
            # record the exact save_dir
            captured["tb_save_dir"] = save_dir
            captured["tb_logger"] = self

    class DummyWBLogger:
        def __init__(self, project, config, version, name, save_dir):
            captured["wb_args"] = dict(
                project=project,
                config=config,
                version=version,
                name=name,
                save_dir=save_dir,
            )
            captured["wb_logger"] = self

        def log_metrics(self, metrics_dict, step):
            # record the logged metrics and step
            captured["wb_logged"] = dict(metrics=metrics_dict, step=step)

    monkeypatch.setattr(
        "src.fl.get_evaluate_fn.TensorBoardLogger", DummyTBLogger
    )
    monkeypatch.setattr(
        "src.fl.get_evaluate_fn.WandbLogger", DummyWBLogger
    )

    # 3) Dummy ServerEvalDataModule
    class DummyDataModule:
        def __init__(self, x_test, y_test, batch_size):
            captured["dm_args"] = dict(
                x_test=x_test, y_test=y_test, batch_size=batch_size
            )

    monkeypatch.setattr(
        "src.fl.get_evaluate_fn.ServerEvalDataModule", DummyDataModule
    )

    # 4) Dummy Trainer
    class DummyTrainer:
        def __init__(self, max_epochs, logger, enable_checkpointing, num_sanity_val_steps):
            captured["trainer_init"] = dict(
                max_epochs=max_epochs, logger=logger, enable_checkpointing=enable_checkpointing, num_sanity_val_steps=num_sanity_val_steps
            )
            self.logger = logger

        def test(self, model, datamodule):
            captured["trainer_test_call"] = dict(
                model=model, datamodule=datamodule
            )
            # return a single‐item list as Lightning does
            return [
                {
                    "test_loss": 0.5,
                    "test_acc": 75.0,
                    "test_f1": 80.0,
                }
            ]

    monkeypatch.setattr(
        "src.fl.get_evaluate_fn.pl.Trainer", DummyTrainer
    )

    return captured


@pytest.fixture
def common_args():
    # synthetic server test set
    x_test_server = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_test_server = np.array([0, 1, 1, 0])
    training_cfg = SimpleNamespace(batch_size=2)

    # dummy eval_model that just records parameters
    eval_model = SimpleNamespace()

    def _set_params(params):
        setattr(eval_model, "params", params)
    eval_model.set_parameters = _set_params

    model_name = "mymodel"
    # build a cfg_base with both loggers configured
    tb_cfg = SimpleNamespace(save_dir="TB_ROOT")
    wb_cfg = SimpleNamespace(project="WB_PROJ", save_dir="WB_ROOT")
    logging = LoggingWrapper(selected_type=None, tb_cfg=tb_cfg, wb_cfg=wb_cfg)
    cfg_base = SimpleNamespace(
        experiment=SimpleNamespace(name="EXP_NAME"),
        logging=logging,
    )

    exp_type = "EXP"
    config_to_add_to_logger = {"foo": "bar"}
    run_dtime = "20250101-000000"

    return dict(
        x_test_server=x_test_server,
        y_test_server=y_test_server,
        training_cfg=training_cfg,
        eval_model=eval_model,
        model_name=model_name,
        cfg_base=cfg_base,
        exp_type=exp_type,
        config_to_add_to_logger=config_to_add_to_logger,
        run_dtime=run_dtime,
    )


def test_tensorboard_branch(patch_dependencies, common_args):
    cap = patch_dependencies
    args = common_args

    # select tensorboard branch
    args["cfg_base"].logging.selected_type = "tensorboard"
    evaluate_fn = get_evaluate_fn(
        x_test_server=args["x_test_server"],
        y_test_server=args["y_test_server"],
        training_cfg=args["training_cfg"],
        eval_model=args["eval_model"],
        model_name=args["model_name"],
        cfg_base=args["cfg_base"],
        exp_type=args["exp_type"],
        config_to_add_to_logger=args["config_to_add_to_logger"],
        run_dtime=args["run_dtime"],
    )

    # call with round=7, dummy parameters, empty config
    loss, metrics = evaluate_fn(7, parameters={"a": 1}, config={})

    # it should have loaded those parameters
    assert args["eval_model"].params == {"a": 1}

    # check the DataModule was constructed correctly
    dm_args = cap["dm_args"]
    np.testing.assert_array_equal(dm_args["x_test"], args["x_test_server"])
    np.testing.assert_array_equal(dm_args["y_test"], args["y_test_server"])
    assert dm_args["batch_size"] == args["training_cfg"].batch_size

    # check the Trainer was constructed with the TB logger
    assert cap["trainer_init"]["max_epochs"] == 1
    assert cap["trainer_init"]["logger"] is cap["tb_logger"]

    # TensorBoardLogger path uses our fixed time
    expected_tb_path = (
        f"TB_ROOT/{args['cfg_base'].experiment.name}/"
        f"FIXEDTIME/{args['exp_type']}_{args['model_name']}/test"
    )
    assert cap["tb_save_dir"] == expected_tb_path

    # test() must have been called on the dummy trainer
    assert cap["trainer_test_call"]["model"] is args["eval_model"]
    # datamodule passed correctly
    assert cap["trainer_test_call"]["datamodule"].__class__.__name__ == "DummyDataModule"

    # return values come from our DummyTrainer.test()
    assert loss == 0.5
    assert metrics == {"accuracy": 75.0, "test_f1": 80.0}

    # TB branch should NOT call log_metrics
    assert "wb_logged" not in cap


def test_wandb_branch(patch_dependencies, common_args):
    cap = patch_dependencies
    args = common_args

    # select wandb branch
    args["cfg_base"].logging.selected_type = "wandb"
    evaluate_fn = get_evaluate_fn(
        x_test_server=args["x_test_server"],
        y_test_server=args["y_test_server"],
        training_cfg=args["training_cfg"],
        eval_model=args["eval_model"],
        model_name=args["model_name"],
        cfg_base=args["cfg_base"],
        exp_type=args["exp_type"],
        config_to_add_to_logger=args["config_to_add_to_logger"],
        run_dtime=args["run_dtime"],
    )

    loss, metrics = evaluate_fn(3, parameters={"b": 2}, config={})

    # parameters still loaded
    assert args["eval_model"].params == {"b": 2}

    # WandbLogger was constructed with correct args
    wb = cap["wb_args"]
    assert wb["project"] == "WB_PROJ"
    assert wb["config"] == args["config_to_add_to_logger"]
    assert wb["version"].startswith(
        f"{args['run_dtime']}_{args['model_name']}_test")
    assert wb["name"] == f"{args['exp_type']}_{args['model_name']}_test"
    assert wb["save_dir"].endswith(
        f"{args['cfg_base'].experiment.name}/{args['exp_type']}_{args['model_name']}_test"
    )

    # test() called on DummyTrainer with WB logger
    assert cap["trainer_init"]["logger"] is cap["wb_logger"]

    # And wandb branch should call log_metrics with full results + round
    assert cap["wb_logged"] == {
        "metrics": {
            "test_loss": 0.5,
            "test_acc": 75.0,
            "test_f1": 80.0,
            "round": 3,
        },
        "step": 3,
    }

    # return values come from DummyTrainer.test()
    assert loss == 0.5
    assert metrics == {"accuracy": 75.0, "test_f1": 80.0}
