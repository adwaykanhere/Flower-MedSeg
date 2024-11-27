# client.py

import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict
import torch
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import json

# Add nnUNet to the Python path
nnunet_path = "/Users/akanhere/Documents/nnUNet"  # Replace with the path to your nnUNet repository
sys.path.append(nnunet_path)

# Correct imports from nnunet
from nnunet.training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results, maybe_mkdir_p

# Argument parsing
parser = argparse.ArgumentParser(description="nnUNet Flower Client")
parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data")
parser.add_argument("--plans-path", type=str, required=True, help="Path to plans.json")
parser.add_argument("--fold", type=int, default=0, help="Fold number")
parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
parser.add_argument("--client-id", type=str, required=True, help="Client ID")
args = parser.parse_args()

class nnUNetClient(fl.client.NumPyClient):
    def __init__(self, trainer):
        self.trainer = trainer
        self.intensity_properties = {}  # For preprocessing rounds

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays
        parameters = [val.cpu().numpy() for _, val in self.trainer.network.state_dict().items()]
        print("Client: Sending parameters to the server.")
        return parameters

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.trainer.network.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.network.load_state_dict(state_dict, strict=True)
        print("Client: Parameters from server have been set.")

    def fit(self, parameters, config):
        current_round = int(config.get("current_round", 0))
        print(f"Client {args.client_id}: Starting fit for round {current_round}")

        if current_round == -2:
            # Preprocessing round
            print("Preprocessing data and collecting intensity properties.")
            self.preprocess_data()
            # Return empty parameters and send intensity properties
            return [], 0, {"intensity_properties": self.intensity_properties}
        elif current_round == -1:
            # Finalize preprocessing with global intensity properties
            print("Finalizing preprocessing with global intensity properties.")
            self.finalize_preprocessing()
            return [], 0, {}
        else:
            # Training round
            if parameters:
                self.set_parameters(parameters)
            else:
                print("No parameters received; starting training from scratch.")
            self.trainer.run_training()
            if config.get("save_model") == "True":
                self.save_model(current_round)
            return self.get_parameters(), len(self.trainer.dataloader_train.dataset), {}

    def evaluate(self, parameters, config):
        current_round = int(config.get("current_round", 0))
        print(f"Client {args.client_id}: Starting evaluation for round {current_round}")
        self.set_parameters(parameters)
        # Evaluate model
        metrics = self.trainer.validate()
        dice_score = metrics["mean_dice"]
        print(f"Client {args.client_id} evaluation Dice score: {dice_score}")
        return 1.0 - dice_score, len(self.trainer.dataloader_val.dataset), {"dice": dice_score}

    def preprocess_data(self):
        # Implement data preprocessing and collect intensity properties
        print("Collecting intensity properties from local data.")
        # Load dataset properties
        dataset_properties_path = os.path.join(args.data_dir, 'dataset_properties.pkl')
        if os.path.exists(dataset_properties_path):
            with open(dataset_properties_path, 'rb') as f:
                dataset_properties = pickle.load(f)
            self.intensity_properties = dataset_properties.get('intensityproperties', {})
            print(f"Collected intensity properties: {self.intensity_properties}")
        else:
            print("Dataset properties file not found.")
            # Implement data preprocessing here if necessary

    def finalize_preprocessing(self):
        # Use global intensity properties to adjust preprocessing
        print("Adjusting preprocessing with global intensity properties.")
        global_intensity_props_path = Path("global_models/global_intensity_properties.json")
        if global_intensity_props_path.exists():
            with open(global_intensity_props_path, "r") as f:
                global_intensity_properties = json.load(f)
            print(f"Loaded global intensity properties: {global_intensity_properties}")
            # Adjust preprocessing steps accordingly
            # For example, normalize data based on global intensity properties
        else:
            print("Global intensity properties not found.")

    def save_model(self, round_num):
        output_folder = self.trainer.output_folder
        model_path = os.path.join(output_folder, f"client_{args.client_id}_model_round_{round_num}.pth")
        torch.save(self.trainer.network.state_dict(), model_path)
        print(f"Client {args.client_id}: Model saved to {model_path}")

def main():
    # Set up the trainer
    plans_file = args.plans_path
    fold = args.fold
    client_id = args.client_id
    output_folder_name = f"nnUNet_{client_id}"

    # Ensure the output directory exists
    output_folder = os.path.join(nnUNet_results, output_folder_name)
    maybe_mkdir_p(output_folder)

    # Load plans (plans are typically stored as JSON in nnUNet v2)
    with open(plans_file, 'r') as f:
        plans = json.load(f)

    # Load dataset JSON
    dataset_json_path = os.path.join(args.data_dir, 'dataset.json')
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    # Initialize the trainer
    trainer = nnUNetTrainerV2(
        plans=plans,
        configuration='3d_fullres',  # or another configuration as per your setup
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    trainer.initialize()

    # Create the Flower client
    client = nnUNetClient(trainer)
    # Start the client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
