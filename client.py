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
import shutil
from multiprocessing import Pool

# Add nnUNet to the Python path
nnunet_path = "/path/to/nnUNet"  # Replace with your nnUNet path
sys.path.append(nnunet_path)

from nnunet.training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.paths import nnUNet_preprocessed, nnUNet_results, maybe_mkdir_p
from nnunet.preprocessing.preprocessing import resample_data_or_seg
from nnunet.utilities.file_endings import save_pickle
from batchgenerators.utilities.file_and_folder_operations import join

# Argument parsing
parser = argparse.ArgumentParser(description="nnUNet Flower Client")
parser.add_argument("--data-dir", type=str, required=True, help="Path to raw data")
parser.add_argument("--preprocessed-dir", type=str, required=True, help="Path to preprocessed data")
parser.add_argument("--plans-path", type=str, required=True, help="Path to plans.json")
parser.add_argument("--fold", type=int, default=0, help="Fold number")
parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
parser.add_argument("--client-id", type=str, required=True, help="Client ID")
args = parser.parse_args()

class nnUNetClient(fl.client.NumPyClient):
    def __init__(self, trainer, data_dir):
        self.trainer = trainer
        self.data_dir = data_dir
        self.preprocessed_output_dir = args.preprocessed_dir
        self.intensity_properties = {}
        self.dataset_fingerprint = {}
        self.plans = trainer.plans_manager.plans
        self.dataset_json = trainer.dataset_json

    def get_parameters(self):
        parameters = [val.cpu().numpy() for _, val in self.trainer.network.state_dict().items()]
        print("Client: Sending parameters to the server.")
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(self.trainer.network.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.network.load_state_dict(state_dict, strict=True)
        print("Client: Parameters from server have been set.")

    def fit(self, parameters, config):
        current_round = int(config.get("current_round", 0))
        print(f"Client {args.client_id}: Starting fit for round {current_round}")

        if current_round == -2:
            print("Preprocessing data and collecting intensity properties.")
            self.preprocess_data()
            return [], 0, {"intensity_properties": self.intensity_properties}
        elif current_round == -1:
            print("Finalizing preprocessing with global intensity properties.")
            self.finalize_preprocessing(config.get("global_intensity_properties"))
            return [], 0, {}
        else:
            if parameters:
                self.set_parameters(parameters)
            self.trainer.run_training()
            self.save_model(current_round)
            return self.get_parameters(), len(self.trainer.dataloader_train.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.trainer.validate()
        dice_score = metrics["mean_dice"]
        print(f"Client {args.client_id} evaluation Dice score: {dice_score}")
        return 1.0 - dice_score, len(self.trainer.dataloader_val.dataset), {"dice": dice_score}

    def preprocess_data(self):
        dataset_json_path = join(self.data_dir, "dataset.json")
        if not os.path.exists(dataset_json_path):
            raise FileNotFoundError(f"{dataset_json_path} not found.")
        with open(dataset_json_path, "r") as f:
            dataset_json = json.load(f)

        voxel_values = {mod_id: [] for mod_id in dataset_json["modality"]}
        for case in dataset_json["training"]:
            img_path = join(self.data_dir, case["image"])
            label_path = join(self.data_dir, case["label"])
            img, label = self.load_nifti(img_path), self.load_nifti(label_path)
            for mod_id, modality_data in enumerate(img):
                foreground = modality_data[label > 0]
                if foreground.size > 0:
                    voxel_values[mod_id].extend(foreground.tolist())

        self.intensity_properties = {
            mod: self.compute_intensity_statistics(voxels) for mod, voxels in voxel_values.items()
        }
        save_json(self.intensity_properties, join(self.preprocessed_output_dir, "intensity_properties.json"))
        print("Saved intensity properties.")

    def finalize_preprocessing(self, global_props):
        for mod_id, props in global_props.items():
            print(f"Normalizing modality {mod_id} using global properties.")
            # Normalization logic can be added here
        finalized_dir = join(self.preprocessed_output_dir, "finalized")
        shutil.copytree(self.preprocessed_output_dir, finalized_dir, dirs_exist_ok=True)

    @staticmethod
    def compute_intensity_statistics(voxels):
        arr = np.array(voxels)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "percentile_00_5": float(np.percentile(arr, 0.5)),
            "percentile_99_5": float(np.percentile(arr, 99.5)),
        }

    def save_model(self, round_num):
        output_folder = self.trainer.output_folder
        model_path = join(output_folder, f"client_{args.client_id}_model_round_{round_num}.pth")
        torch.save(self.trainer.network.state_dict(), model_path)
        print(f"Client {args.client_id}: Model saved to {model_path}")

    def load_nifti(self, filepath):
        import nibabel as nib
        return nib.load(filepath).get_fdata()

def main():
    plans_file = args.plans_path
    fold = args.fold
    output_folder_name = f"nnUNet_{args.client_id}"
    output_folder = join(nnUNet_results, output_folder_name)
    maybe_mkdir_p(output_folder)

    with open(plans_file, "r") as f:
        plans = json.load(f)
    dataset_json_path = join(args.data_dir, "dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    trainer = nnUNetTrainerV2(
        plans=plans,
        configuration="3d_fullres",
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    client = nnUNetClient(trainer, args.data_dir)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
