"""Inference script."""

import argparse
import logging

import numpy as np
import torch
import xarray as xr

from src.transforms import load_transforms
from src.data import UKCPDataset
from src.networks import ADM
from src.distributions import DistributionSampler
from src.solvers import EulerSolver, HeunSolver
from src.config import load_config

from common.logging.mlflow import setup_mlflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train network.")

    parser.add_argument(
        "--solver_type",
        type=str,
        help="Name of numerical solver to use. Expected to be 'euler' or 'heun'.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        help="Number of time steps to simulate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Exponent for the time-stepping scheme.",
    )
    parser.add_argument(
        "--config_filename",
        type=str,
        help="Name of configuration file used to train the model. Expected to be in src/configs.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--transforms_path",
        type=str,
        help="Path to transforms file. Fit during training.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test dataset file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to directory to save the model predictions.",
    )

    args = parser.parse_args()

    if args.solver_type not in ["euler", "heun"]:
        raise ValueError(f"Expected solver to be 'euler' or 'heun', but got {args.solver_type}.")

    return args


@setup_mlflow
def main(
    solver_type: str,
    n_steps: int,
    alpha: float,
    config_filename: str,
    model_path: str,
    transforms_path: str,
    test_data_path: str,
    batch_size: int,
    output_path: str,
):
    config = load_config(config_filename)
    targets: list[str] = list(config["targets"].keys())  # list of target variable names
    logger.info(f"Configuration:\n{config}")

    transforms = load_transforms(transforms_path)
    dataset = UKCPDataset(data_path=test_data_path, config=config, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ADM(
        num_conditioning_variables=len(config["predictors"]),
        num_target_variables=len(config["targets"]),
        **config["model"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    coupled = config["source"]["type"] == "coupled"
    if not coupled:
        source_sampler = DistributionSampler(config, DEVICE)
    else:
        # Source data will be returned from the dataset when using coupled source/target.
        source_sampler = None

    if solver_type == "euler":
        solver = EulerSolver(n_steps=n_steps, alpha=alpha, model=model, device=DEVICE)
    else:
        solver = HeunSolver(n_steps=n_steps, alpha=alpha, model=model, device=DEVICE)

    predictions: list[torch.Tensor] = []
    for batch in dataloader:
        if coupled:
            conditioning, _, x_0 = batch
        else:
            conditioning, _, _ = batch
            x_0 = source_sampler.sample(
                (
                    conditioning.shape[0],
                    len(config["targets"]),
                    conditioning.shape[2],
                    conditioning.shape[3],
                )
            )  # shape (batch_size, num_target_variables, lat, lon)

        conditioning = conditioning.to(DEVICE)
        x_0 = x_0.to(DEVICE)

        model.eval()
        with torch.no_grad():
            x_1 = solver.simulate_ode(
                x_0=x_0, conditioning=conditioning
            )  # shape (batch_size, num_target_variables, lat, lon)
            predictions.append(x_1.cpu())

    predictions_ar = torch.cat(
        predictions, dim=0
    ).numpy()  # shape (time, num_target_variables, lat, lon)

    # Inverse any transforms applied to the targets.
    for idx, target in enumerate(targets):
        target_transform = transforms["targets"][target]
        predictions_ar[:, idx, :, :] = target_transform.inverse_transform(
            predictions_ar[:, idx, :, :]
        )

    targets_ar = (
        dataset.ds[targets]
        .to_array("variable")
        .transpose("time", "variable", "grid_latitude", "grid_longitude")
    )

    ds = xr.Dataset(
        {
            "predictions": (
                ("time", "variable", "grid_latitude", "grid_longitude"),
                predictions_ar,
                {
                    "grid_mapping": "rotated_latitude_longitude",
                },
            ),
            "targets": (
                ("time", "variable", "grid_latitude", "grid_longitude"),
                targets_ar.values,
                {
                    "grid_mapping": "rotated_latitude_longitude",
                },
            ),
            "rotated_latitude_longitude": dataset.ds["rotated_latitude_longitude"],
            "grid_latitude_bnds": dataset.ds["grid_latitude_bnds"],
            "grid_longitude_bnds": dataset.ds["grid_longitude_bnds"],
        },
        coords={
            "time": dataset.ds["time"].values,
            "variable": targets,
            "grid_latitude": dataset.ds["grid_latitude"].values,
            "grid_longitude": dataset.ds["grid_longitude"].values,
        },
    )

    ds.attrs["solver_type"] = solver_type
    ds.attrs["n_steps"] = n_steps
    ds.attrs["alpha"] = alpha

    output_file = f"{output_path}/predictions.nc"
    ds.to_netcdf(output_file, format="NETCDF4", engine="netcdf4")

    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(
        solver_type=args.solver_type,
        n_steps=args.n_steps,
        alpha=args.alpha,
        config_filename=args.config_filename,
        model_path=args.model_path,
        transforms_path=args.transforms_path,
        test_data_path=args.test_data_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
