import numpy as np
from typing import List, Tuple, Optional, Dict

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.typing import FitRes


class FedDyn(FedAvg):
    """FedDyn strategy (A Dynamic Regularization for Federated Learning)."""

    def __init__(
        self,
        *,
        rho: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rho = rho
        # Will hold the previous global parameters (w_t)
        self.prev_global: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # 1) Get the vanilla FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_params is None:
            return None, metrics

        # 2) Convert to numpy arrays
        current_ndarrays = parameters_to_ndarrays(aggregated_params)

        # 3) If this is the first round, just use the average as-is
        if self.prev_global is None:
            adjusted_ndarrays = current_ndarrays
        else:
            # 4) Apply dynamic regularization:
            #    w_{t+1} = w̄_{t+1} - rho * (w̄_{t+1} - w_t)
            adjusted_ndarrays = [
                w_bar - self.rho * (w_bar - w_prev)
                for w_bar, w_prev in zip(current_ndarrays, self.prev_global)
            ]

        # 5) Store for the next round
        self.prev_global = adjusted_ndarrays

        # 6) Convert back to Parameters and return
        new_parameters = ndarrays_to_parameters(adjusted_ndarrays)
        return new_parameters, metrics


class FedBN(FedAvg):
    """FedBN strategy: only aggregate non-BN layers, keep BatchNorm locally."""

    def __init__(
        self,
        *,
        parameter_names: List[str],
        **kwargs,
    ) -> None:
        """
        Args:
            parameter_names: List of parameter keys in the exact order used
                             by flwr.common.parameters_to_ndarrays / 
                             ndarrays_to_parameters.  E.g., list(model.state_dict().keys()).
        """
        super().__init__(**kwargs)
        self.parameter_names = parameter_names
        # Will hold the previous global parameters
        self.prev_global: Optional[List[np.ndarray]] = None
        self.bn_indices = [
            i for i, name in enumerate(self.parameter_names)
            if "bn" in name.lower()
        ]

    # def configure_fit(self, server_round, parameters, client_manager):
    #     # parameters: a flwr.common.Parameters proto
    #     ndarrays = parameters_to_ndarrays(parameters)

    #     # Zero-out (or leave unchanged) the BN slots:
    #     for idx in self.bn_indices:
    #         # Option A: zero them (clients will keep their own running stats)
    #         ndarrays[idx] = np.zeros_like(ndarrays[idx])
    #         # Option B: leave them untouched so clients just keep what they had:
    #         # ndarrays[idx] = ndarrays[idx]  # no-op

    #     new_parameters = ndarrays_to_parameters(ndarrays)

    #     # The rest is identical to FedAvg
    #     fit_ins = super().configure_fit(server_round, new_parameters, client_manager)
    #     return fit_ins

    # def aggregate_fit(
    #     self,
    #     rnd: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[BaseException],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     # 1) Run standard FedAvg aggregation
    #     aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)
    #     if aggregated_params is None:
    #         return None, metrics

    #     # 2) Convert to numpy arrays
    #     current_ndarrays = parameters_to_ndarrays(aggregated_params)

    #     # 3) On first round, just take the aggregated
    #     if self.prev_global is None:
    #         new_ndarrays = current_ndarrays
    #     else:
    #         new_ndarrays = []
    #         # 4) For each parameter, if it's a BN layer, keep previous global
    #         for name, w_bar, w_prev in zip(
    #             self.parameter_names, current_ndarrays, self.prev_global
    #         ):
    #             if "bn" in name.lower():
    #                 # Keep previous BN weights/biases
    #                 new_ndarrays.append(w_prev)
    #             else:
    #                 # Aggregate normally
    #                 new_ndarrays.append(w_bar)

    #     # 5) Store for next round
    #     self.prev_global = new_ndarrays

    #     # 6) Convert back and return
    #     new_parameters = ndarrays_to_parameters(new_ndarrays)
    #     return new_parameters, metrics
