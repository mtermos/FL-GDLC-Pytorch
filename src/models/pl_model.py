import os
import json
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from src.utils import NumpyEncoder, plot_confusion_matrix, calculate_fpr_fnr_with_global
from src.models.loss_functions import FocalLoss


class LitClassifier(pl.LightningModule):
    def __init__(self, model, model_name, training_cfg, labels_mapping, weight_tensor=None, using_wandb=False):

        super().__init__()
        # self.save_hyperparameters()

        self.model = model
        self.model_name = model_name
        self.learning_rate = training_cfg.learning_rate
        self.weight_decay = training_cfg.weight_decay
        self.labels = list(labels_mapping.values())
        self.labels_mapping = labels_mapping
        self.using_wandb = using_wandb
        self.multi_class = training_cfg.multi_class
        self.batch_size = training_cfg.batch_size
        self.server_round = 0

        if training_cfg.optimizer == "adam":
            self.optimizer = Adam
        elif training_cfg.optimizer == "sgd":
            self.optimizer = SGD

        if training_cfg.loss_type == "focal":
            # alpha = weight_tensor
            # alpha = weight_tensor / weight_tensor.sum()
            # print(f"==>> alpha: {alpha}")
            # self.criterion = FocalLoss(alpha=alpha,
            self.criterion = FocalLoss(alpha=training_cfg.focal_loss_alpha,
                                       gamma=training_cfg.focal_loss_gamma, reduction=training_cfg.focal_loss_reduction)
        elif training_cfg.loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.train_epoch_metrics = {}
        self.val_epoch_metrics = {}
        self.train_outputs = {"preds": [], "targets": []}
        self.val_outputs = {"preds": [], "targets": []}
        self.test_outputs = {"preds": [], "targets": []}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 1:
            unique, counts = np.unique(y.numpy(), return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"Batch {batch_idx}: {dist}")

        pred = self(x)
        loss = self.criterion(pred, y)
        pred = pred.argmax(dim=1)
        acc = (pred == y).float().mean() * 100.0
        self.log('train_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.train_outputs["preds"].append(pred)
        self.train_outputs["targets"].append(y)

        return loss

    def on_train_epoch_end(self):
        all_preds = th.cat(self.train_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.train_outputs["targets"]).detach().cpu().numpy()
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("train_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.train_outputs = {"preds": [], "targets": []}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = pred.argmax(dim=1)
        acc = (pred == y).float().mean() * 100.0
        self.log('val_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.val_outputs["preds"].append(pred)
        self.val_outputs["targets"].append(y)

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "sanity_checking", False):
            self.val_outputs = {"preds": [], "targets": []}
            return  # skip any summary/logging during the sanity‐check
        all_preds = th.cat(self.val_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.val_outputs["targets"]).detach().cpu().numpy()
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("val_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)

        print("Validation Classification Report:\n", report)

        if self.using_wandb:
            class_report = classification_report(all_targets, all_preds,
                                                 digits=4,
                                                 output_dict=True,
                                                 zero_division=0)
            report_df = pd.DataFrame(class_report).T.reset_index()
            report_df = report_df.rename(columns={"index": "class"})

            table = wandb.Table(dataframe=report_df)
            wandb.log({f"classification_report_{self.server_round}": table})
        # columns = ["class", "precision", "recall", "f1-score", "support"]
        # data = [
        #     [row["class"],
        #     row["precision"],
        #     row["recall"],
        #     row["f1-score"],
        #     int(row["support"])]
        #     for _, row in report_df.iterrows()
        # ]
        # table2 = wandb.Table(data=data, columns=columns)
        # wandb.log({"classification_report_manual": table2})
        # report_columns =  ["Class", "Precision", "Recall", "F1-score", "Support"]
        # class_report = classification_report(all_targets,all_preds).splitlines()

        # report_table = []
        # for line in class_report[2:(len(self.labels)+2)]:
        #     report_table.append(line.split())

        # wandb.log({
        #     "Confusion Matix": wandb.plot.confusion_matrix(y_true=all_targets, preds=all_preds, class_names=self.labels),
        #     "Classification Report": wandb.Table(data=report_table, columns=report_columns)
        #     })
        self.val_outputs = {"preds": [], "targets": []}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = pred.argmax(dim=1)
        acc = (pred == y).float().mean() * 100.0

        self.log("test_loss", loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.test_outputs["preds"].append(pred)
        self.test_outputs["targets"].append(y)
        return {"test_loss": loss, "test_acc": acc, "preds": pred, "targets": y}

    def on_test_epoch_end(self):
        all_preds = th.cat(self.test_outputs["preds"]).detach().cpu().numpy()
        all_targets = th.cat(
            self.test_outputs["targets"]).detach().cpu().numpy()
        self.test_outputs = {"preds": [], "targets": []}
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("test_f1", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        all_targets = np.vectorize(self.labels_mapping.get)(all_targets)
        all_preds = np.vectorize(self.labels_mapping.get)(all_preds)

        cm = confusion_matrix(all_targets, all_preds, labels=self.labels)

        cr = classification_report(
            all_targets, all_preds, digits=4, output_dict=True, zero_division=0)
        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100

        results_fpr_fnr = calculate_fpr_fnr_with_global(cm)
        fpr = results_fpr_fnr["global"]["FPR"]
        fnr = results_fpr_fnr["global"]["FNR"]

        results = {
            "test_weighted_f1": weighted_f1,
            "test_fpr": fpr,
            "test_fnr": fnr,
            "classification_report": cr,
            "results_fpr_fnr": results_fpr_fnr
        }

        os.makedirs("temp", exist_ok=True)
        json_path = os.path.join("temp", f"{self.model_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        if self.using_wandb:
            wandb.save(json_path)

        print("=== Test Evaluation Metrics ===")
        print("Classification Report:\n", report)

        # cm_normalized = confusion_matrix(
        #     all_targets, all_preds, labels=self.labels, normalize="true")
        # fig = plot_confusion_matrix(cm=cm,
        #                             normalized=False,
        #                             target_names=self.labels,
        #                             title=f"Confusion Matrix of {self.model_name}",
        #                             file_path=None,
        #                             show_figure=False)

        # if self.using_wandb:
        #     wandb.log({f"confusion_matrix_{self.model_name}": wandb.Image(
        #         fig), "epoch": self.current_epoch})
        # fig = plot_confusion_matrix(cm=cm_normalized,
        #                             normalized=True,
        #                             target_names=self.labels,
        #                             title=f"Confusion Matrix of {self.model_name}",
        #                             file_path=None,
        #                             show_figure=False)

        # if self.using_wandb:
        #     wandb.log({f"confusion_matrix_{self.model_name}_normalized": wandb.Image(
        #         fig), "epoch": self.current_epoch})

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: th.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
