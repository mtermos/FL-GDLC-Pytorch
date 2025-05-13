# import os
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pytest
# from pathlib import Path

# from src.utils import NumpyEncoder
# from src.models.pl_model import LitClassifier


# class DummyCfg:
#     def __init__(self, optimizer="adam", multi_class=False, batch_size=4, loss_type="cross_entropy"):
#         self.learning_rate = 0.01
#         self.weight_decay = 0.0
#         self.optimizer = optimizer
#         self.multi_class = multi_class
#         self.batch_size = batch_size
#         self.loss_type = loss_type


# class TestLitClassifier(LitClassifier):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # capture logs here
#         self.logged = {}

#     def log(self, name, value, **kwargs):
#         # record only the numeric value
#         self.logged[name] = float(value.detach().cpu().item()) if hasattr(
#             value, 'detach') else float(value)


# @pytest.fixture
# def classifier(tmp_path):
#     # simple 2→2 linear model
#     model = nn.Linear(2, 2, bias=False)
#     # map classes 0 and 1 to names
#     labels_mapping = {0: "class0", 1: "class1"}
#     # equal weights
#     weight_tensor = torch.ones(2)
#     cfg = DummyCfg(optimizer="adam", multi_class=False,
#                    batch_size=4, loss_type="cross_entropy")
#     lit = TestLitClassifier(model, model_name="test_model", training_cfg=cfg,
#                             labels_mapping=labels_mapping, weight_tensor=weight_tensor,
#                             using_wandb=False)
#     # ensure tmp_path is the cwd so JSON writes there
#     os.chdir(tmp_path)
#     return lit


# def make_batch(n=4):
#     # Generate a batch of size n, 2-dim input, half class 0 half class 1
#     x = torch.randn(n, 2)
#     y = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
#     return x, y


# def test_forward_and_configure_optimizers(classifier):
#     x = torch.randn(3, 2)
#     out = classifier(x)
#     # should equal the underlying model
#     assert torch.allclose(out, classifier.model(x))

#     opt = classifier.configure_optimizers()
#     assert isinstance(opt, optim.Adam)
#     # check hyperparams
#     assert pytest.approx(
#         opt.param_groups[0]["lr"], rel=1e-6) == classifier.learning_rate
#     assert pytest.approx(
#         opt.param_groups[0]["weight_decay"], rel=1e-6) == classifier.weight_decay


# def test_training_and_on_train_epoch_end(classifier):
#     x, y = make_batch()
#     loss = classifier.training_step((x, y), batch_idx=0)
#     # loss is a scalar tensor
#     assert isinstance(loss, torch.Tensor) and loss.dim() == 0
#     # train_outputs should have one entry
#     assert len(classifier.train_outputs["preds"]) == 1
#     assert len(classifier.train_outputs["targets"]) == 1

#     # call epoch end and check f1 logged ~100%
#     classifier.on_train_epoch_end()
#     assert "train_f1_score" in classifier.logged
#     # assert pytest.approx(
#     #     classifier.logged["train_f1_score"], rel=1e-3) == 100.0
#     # ensure buffers reset
#     assert classifier.train_outputs["preds"] == []
#     assert classifier.train_outputs["targets"] == []


# def test_validation_and_on_validation_epoch_end(classifier):
#     x, y = make_batch()
#     out = classifier.validation_step((x, y), batch_idx=0)
#     # should return dict with val_loss and val_acc
#     assert "val_loss" in out and "val_acc" in out
#     assert len(classifier.val_outputs["preds"]) == 1

#     classifier.on_validation_epoch_end()
#     assert "val_f1_score" in classifier.logged
#     # assert pytest.approx(classifier.logged["val_f1_score"], rel=1e-3) == 100.0
#     assert classifier.val_outputs["preds"] == []


# def test_test_and_on_test_epoch_end_creates_json(classifier, tmp_path):
#     x, y = make_batch()
#     out = classifier.test_step((x, y), batch_idx=0)
#     assert "test_loss" in out and "test_acc" in out and "preds" in out and "targets" in out
#     assert len(classifier.test_outputs["preds"]) == 1

#     # ensure no file yet
#     result_path = Path("temp") / "test_model_results.json"
#     assert not result_path.exists()

#     classifier.on_test_epoch_end()
#     # JSON should be created
#     assert result_path.exists()

#     data = json.loads(result_path.read_text())
#     # check presence of keys
#     for k in ("test_weighted_f1", "test_fpr", "test_fnr", "classification_report", "results_fpr_fnr"):
#         assert k in data

#     # all preds/targets buffers reset
#     assert classifier.test_outputs["preds"] == []
#     assert classifier.test_outputs["targets"] == []


# def test_get_and_set_parameters(classifier):
#     orig = classifier.get_parameters()
#     # shift all by +1
#     new = [arr + 1 for arr in orig]
#     classifier.set_parameters(new)
#     loaded = classifier.get_parameters()
#     # compare elementwise
#     for a, b in zip(new, loaded):
#         assert pytest.approx(a) == b
