"""Training script."""

import argparse
import logging
import yaml
from copy import deepcopy

import torch
import xarray as xr

from common.logging.mlflow import setup_mlflow
from src.transforms import fit_transforms, save_transforms
from src.data import UKCPDataset
from src.scheduler import SCHEDULER_REGISTRY
from src.distributions import DistributionSampler
from src.paths import AffineProbabilityPath, CondOTPath
from src.networks import ADM
from src.train import train_loop, val_loop
from src.config import load_config


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
        "--config_filename",
        type=str,
        help="Name of configuration file. Expected to be in src/configs.",
    )
    parser.add_argument(
        "--training_data_path",
        type=str,
        help="Path to training dataset file.",
    )
    parser.add_argument("--validation_data_path", type=str, help="Path to validation dataset file.")
    parser.add_argument(
        "--outputs_dir_path", type=str, help="Path to directory to write outputs to."
    )

    args = parser.parse_args()
    return args


@setup_mlflow
def main(
    config_filename: str,
    training_data_path: str,
    validation_data_path: str,
    outputs_dir_path: str,
):
    config = load_config(config_filename)
    logger.info(f"Configuration:\n{yaml.dump(config)}")

    # Fit (normalisation) transforms on training dataset.
    with xr.open_dataset(
        training_data_path, chunks={"time": 1000, "grid_latitude": 64, "grid_longitude": 64}
    ) as ds:
        transforms = fit_transforms(config, ds)
        save_transforms(transforms, f"{outputs_dir_path}/transforms.json")

    training_dataset = UKCPDataset(
        data_path=training_data_path, config=config, transforms=transforms
    )

    validation_dataset = UKCPDataset(
        data_path=validation_data_path, config=config, transforms=transforms
    )

    batch_size = config["training"]["batch_size"]
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    scheduler_config = config["scheduler"]
    scheduler_type = scheduler_config["type"]
    scheduler = SCHEDULER_REGISTRY[scheduler_type](**scheduler_config)

    if scheduler_type == "condot":
        probability_path = CondOTPath(scheduler)
    else:
        probability_path = AffineProbabilityPath(scheduler)

    coupled = config["source"]["type"] == "coupled"
    if not coupled:
        source_sampler = DistributionSampler(config, DEVICE)
    else:
        source_sampler = None

    model = ADM(
        num_conditioning_variables=len(config["predictors"]),
        num_target_variables=len(config["targets"]),
        **config["model"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    epochs = config["training"]["epochs"]

    best_val_loss = torch.inf
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(
            dataloader=training_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            coupled=coupled,
            source_sampler=source_sampler,
            probability_path=probability_path,
        )

        val_loss = val_loop(
            dataloader=validation_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=DEVICE,
            epoch=epoch,
            coupled=coupled,
            source_sampler=source_sampler,
            probability_path=probability_path,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())

    model_path = f"{outputs_dir_path}/model.pth"
    logger.info(f"Training complete. Saving model to {model_path}")
    torch.save(best_model_state, model_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        config_filename=args.config_filename,
        training_data_path=args.training_data_path,
        validation_data_path=args.validation_data_path,
        outputs_dir_path=args.outputs_dir_path,
    )
