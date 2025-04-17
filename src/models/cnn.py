import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torchmetrics import Accuracy


class CNN(pl.LightningModule):
    def __init__(self, training_cfg, model_cfg, num_features, num_classes, model_name="mlp"):
        """
        Args:
            hidden_units (list): A list of integers specifying the number of neurons in each hidden layer.
                                 e.g., [128, 64] -> two layers with 128 and 64 neurons, respectively.
            num_features  (int): Number of input features per sample.
            num_classes   (int): Number of classes (for classification).
            use_bn       (bool): Whether to use Batch Normalization after each hidden layer.
            dropout     (float): Dropout probability in the classifier.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = training_cfg.learning_rate
        if training_cfg.optimizer == "adam":
            self.optimizer = Adam
        elif training_cfg.optimizer == "sgd":
            self.optimizer = SGD

        if model_cfg.dense.activation == "relu":
            self.activation = nn.ReLU()
        elif model_cfg.dense.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()

        layers = []
        input_dim = num_features

        for hidden_dim in model_cfg.dense.units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            if model_cfg.dense.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if model_cfg.dense.dropout:
                layers.append(nn.Dropout(model_cfg.dense.dropout_rate))
            input_dim = hidden_dim

        # Final layer for classification
        if training_cfg.multi_class:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Metrics
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.train_accuracy(y_hat, y)
        self.log("train_acc", self.train_accuracy,
                 on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy(y_hat, y)
        self.log("val_acc", self.val_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.test_accuracy(y_hat, y)
        self.log("test_acc", self.test_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: th.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
