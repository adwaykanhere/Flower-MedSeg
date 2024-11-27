# server.py

import os
from pathlib import Path
import shutil
import json
import math
from collections import OrderedDict
import torch
import flwr as fl
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
    EvaluateRes,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Optional, Union

# Custom strategy to handle preprocessing and aggregation
class CustomAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        save_global_path: str,
        total_rounds: int,
        aggregation_strategy: str = "fedavg",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.best_metric = -float("inf")
        self.save_global_path = Path(save_global_path)
        self.total_rounds = total_rounds
        self.aggregation_strategy = aggregation_strategy
        self.parameters = None
        self.intensity_properties = {}
        self.federated_round = 0

        self.save_global_path.mkdir(exist_ok=True, parents=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ):
        self.federated_round = server_round
        if server_round == -2:
            # Preprocessing round: collect data fingerprints
            print("Preprocessing round!")
            self.collect_intensity_properties(results)
            # Prepare for next round
            return None, {}
        elif server_round == -1:
            # Finalize preprocessing and prepare for training
            print("Finalizing preprocessing!")
            self.finalize_preprocessing()
            return None, {}
        else:
            # Training rounds: aggregate model updates
            if self.aggregation_strategy == "fedavg":
                aggregated_parameters = self.fed_avg(results)
            elif self.aggregation_strategy == "feddc":
                aggregated_parameters = self.fed_dc(results, server_round)
            else:
                raise ValueError("Unsupported aggregation strategy.")

            # Save global model
            self.parameters = ndarrays_to_parameters(aggregated_parameters)
            self.save_model(server_round)

            return self.parameters, {}

    def collect_intensity_properties(self, results):
        print("Collecting intensity properties from clients.")
        for _, fit_res in results:
            # Assuming clients send intensity properties in `fit_res.metrics`
            client_properties = fit_res.metrics.get("intensity_properties", {})
            for mod_id, properties in client_properties.items():
                if mod_id not in self.intensity_properties:
                    self.intensity_properties[mod_id] = []
                self.intensity_properties[mod_id].append(properties)
        print("Collected intensity properties from all clients.")

    def finalize_preprocessing(self):
        print("Finalizing preprocessing by estimating global intensity properties.")
        global_intensity_properties = self.estimate_global_intensity_properties(
            self.intensity_properties
        )
        # Save global intensity properties to a file
        intensity_props_path = self.save_global_path / "global_intensity_properties.json"
        with open(intensity_props_path, "w") as f:
            json.dump(global_intensity_properties, f)
        print(f"Saved global intensity properties to {intensity_props_path}")

    def estimate_global_intensity_properties(self, intensities_per_mod):
        results = {}
        for mod_id, all_intensities in intensities_per_mod.items():
            total_mean = 0.0
            total_variance = 0.0
            total_samples = 0
            glob_min = float("inf")
            glob_max = float("-inf")
            for stats in all_intensities:
                mean = stats["mean"]
                std = stats["std"]
                n = stats["n"]

                total_mean += mean * n
                total_samples += n
                total_variance += n * (std**2 + (mean - (total_mean / total_samples)) ** 2)
                glob_min = min(glob_min, stats["min"])
                glob_max = max(glob_max, stats["max"])

            glob_mean = total_mean / total_samples
            glob_std = math.sqrt(total_variance / total_samples)

            results[int(mod_id)] = {
                "min": glob_min,
                "max": glob_max,
                "mean": glob_mean,
                "std": glob_std,
            }
        return results

    def fed_avg(self, results):
        print("Performing FedAvg aggregation.")
        total_weight = 0
        base = [0] * len(parameters_to_ndarrays(results[0][1].parameters))
        for _, fit_res in results:
            weight = fit_res.num_examples
            total_weight += weight
            params = parameters_to_ndarrays(fit_res.parameters)
            for i, param in enumerate(params):
                base[i] += param * weight
        aggregated_params = [param / total_weight for param in base]
        return aggregated_params

    def fed_dc(self, results, server_round):
        print("Performing FedDC aggregation (placeholder).")
        # Implement FedDC aggregation logic here
        # For simplicity, we'll use FedAvg as a placeholder
        return self.fed_avg(results)

    def save_model(self, server_round):
        print(f"Saving global model at round {server_round}.")
        ndarrays = parameters_to_ndarrays(self.parameters)
        state_dict = {f"param_{i}": torch.tensor(ndarray) for i, ndarray in enumerate(ndarrays)}
        filename = self.save_global_path / f"global_model_round_{server_round}.pth"
        torch.save(state_dict, filename)
        print(f"Saved global model to {filename}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Aggregate evaluation metrics
        if not results:
            return None, {}
        metrics = [res.metrics["dice"] * res.num_examples for _, res in results]
        examples = [res.num_examples for _, res in results]
        aggregated_metric = sum(metrics) / sum(examples)
        print(f"Round {server_round} aggregated Dice score: {aggregated_metric}")
        # Save model if performance improved
        if aggregated_metric > self.best_metric:
            print(f"New best Dice score: {aggregated_metric}, saving global model.")
            self.best_metric = aggregated_metric
            self.save_model(server_round)
        return super().aggregate_evaluate(server_round, results, failures)

def main():
    # Define the strategy
    strategy = CustomAggregationStrategy(
        save_global_path="global_models",
        total_rounds=5,
        aggregation_strategy="fedavg",  # or "feddc"
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        fraction_eval=1.0,
        min_eval_clients=2,
    )
    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
