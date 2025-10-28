"""Training and validation loops."""

import logging

import torch
import mlflow

from src.distributions import DistributionSampler
from src.paths import AffineProbabilityPath


logger = logging.getLogger(__name__)


def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    coupled: bool,
    source_sampler: DistributionSampler | None,
    probability_path: AffineProbabilityPath,
):
    """Training loop for a single epoch.

    Runs the training loop for a single epoch. Metrics are logged to MLflow.

    Args:
        dataloader: DataLoader for the training data.
        model: The model to train.
        loss_fn: The loss function.
        optimizer: The optimizer.
        device: The device to run the training on.
        epoch: The current epoch.
        coupled: Whether the source distribution is coupled with the target distribution (e.g.,
            low resolution precip. and high resolution precip.). If false, x_0 is sampled from a
            source distribution (e.g., Gaussian(0, 1)) via source_sampler.
        source_sampler: DistributionSampler instance to sample x_0 from source distribution.
            If coupled is false, then this is required. Otherwise, it should be None.
        probability_path: Probability path to sample x_t and the conditional vector field.
    """
    if coupled:
        if source_sampler is not None:
            raise ValueError(
                "source_sampler is not required (and should be None) when coupled is True."
            )
    else:
        if source_sampler is None:
            raise ValueError("source_sampler is required when coupled is False.")

    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()

    for batch_idx, batch in enumerate(dataloader):
        if coupled:
            conditioning, x_1, x_0 = batch
        else:
            conditioning, x_1, _ = batch
            x_0 = source_sampler.sample(x_1.shape)

        conditioning = conditioning.to(device)
        x_1 = x_1.to(device)
        x_0 = x_0.to(device)
        batch_size = x_1.shape[0]
        # TODO: Make sampling of t configurable, instead of always Unif(0, 1)
        t = torch.rand(batch_size, 1, device=device)
        x_t = probability_path.sample(x_0, x_1, t)
        v_t = probability_path.compute_vector_field(x_0, x_1, t)

        v_t_theta = model(x_t, t, conditioning)

        optimizer.zero_grad()

        loss = loss_fn(v_t_theta, v_t)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss_value = loss.item()
            current = (batch_idx * dataloader.batch_size) + batch_size
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]")
            batch_number = (epoch * num_batches) + batch_idx
            mlflow.log_metric("Training loss", loss_value, step=batch_number)


def val_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    coupled: bool,
    source_sampler: DistributionSampler | None,
    probability_path: AffineProbabilityPath,
):
    if coupled:
        if source_sampler is not None:
            raise ValueError(
                "source_sampler is not required (and should be None) when coupled is True."
            )
    else:
        if source_sampler is None:
            raise ValueError("source_sampler is required when coupled is False.")

    model.eval()
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            if coupled:
                conditioning, x_1, x_0 = batch
            else:
                conditioning, x_1, _ = batch
                x_0 = source_sampler.sample(x_1.shape)

            conditioning = conditioning.to(device)
            x_1 = x_1.to(device)
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            t = torch.rand(batch_size, 1, device=device)
            x_t = probability_path.sample(x_0, x_1, t)
            v_t = probability_path.compute_vector_field(x_0, x_1, t)

            v_t_theta = model(x_t, t, conditioning)

            loss = loss_fn(v_t_theta, v_t)
            loss = loss.item()

            val_loss += loss

    val_loss /= num_batches

    logger.info(f"\nValidation loss: {val_loss:>7f}\n")
    mlflow.log_metric("Validation loss", val_loss, step=epoch + 1)

    return val_loss
