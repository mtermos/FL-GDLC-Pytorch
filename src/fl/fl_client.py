import flwr as fl
import pytorch_lightning as pl
import wandb
import torch


class FLClient(fl.client.NumPyClient):
    def __init__(self, data_module, model, logger, logger_type):
        self.data_module = data_module
        self.model = model
        self.logger = logger
        self.logger_type = logger_type

        # Setup data module
        self.data_module.setup()

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=1,  # Will be updated in fit()
            logger=self.logger,
            enable_progress_bar=False
        )

    def get_parameters(self, config):
        return self.model.get_parameters()
        # return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = {k: torch.tensor(v) for k, v in params_dict}
        # self.model.load_state_dict(state_dict, strict=True)
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters)

        # Update learning rate if provided
        if 'lr' in config:
            self.model.alpha = float(config['lr'])

        # Update number of epochs
        # self.trainer.max_epochs = config['local_epochs']

        # Train the model
        self.trainer.fit(self.model, self.data_module)

        # Get updated parameters
        parameters_prime = self.get_parameters({})
        num_examples = len(self.data_module.train_dataset)

        # Get metrics
        metrics = {
            'train_loss': float(self.trainer.callback_metrics.get('train_loss', 0.0)),
            'train_f1_score': float(self.trainer.callback_metrics.get('train_f1_score', 0.0)),
            'val_loss': float(self.trainer.callback_metrics.get('val_loss', 0.0)),
            'val_acc': float(self.trainer.callback_metrics.get('val_acc', 0.0)),
            'val_f1_score': float(self.trainer.callback_metrics.get('val_f1_score', 0.0))
        }

        if 'server_round' in config:
            server_round = float(config['server_round'])

            metrics["round"] = server_round
            metrics["lr"] = float(config['lr'])

            if self.logger_type == "wandb":
                self.logger.log_metrics(metrics, step=server_round)

        return parameters_prime, num_examples, metrics

    def evaluate(self, parameters, config):
        if not getattr(self.data_module, "do_validate", True):
            # Return dummy values: no loss, zero examples, empty metrics
            return 0.0, 0, {}

        # Set model parameters
        self.set_parameters(parameters)

        # Evaluate the model
        results = self.trainer.validate(self.model, self.data_module)

        # Extract metrics
        loss = float(results[0]['val_loss'])
        accuracy = float(results[0]['val_acc'])
        f1s = float(results[0]['val_f1_score'])
        num_examples = len(self.data_module.val_dataset)

        wandb.finish(quiet=True)
        return loss, num_examples, {"loss": loss, "accuracy": accuracy, "f1s": f1s}
