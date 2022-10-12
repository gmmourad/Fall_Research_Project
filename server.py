import flwr as fl
import sys as sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
                print(f"Saving round {rnd} aggregated_weights . . .")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

strategy = SaveModelStrategy()


fl.server.start_server(
    server_address = "0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3)
)
