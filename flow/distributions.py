import torch


class DistributionSampler:
    """Sample from a distribution."""

    def __init__(self, config: dict, device: torch.device):
        """Initialise.

        Args:
            config: Configuration dictionary containing source type and parameters.
            device: Device to perform computations on (e.g., 'cpu' or 'cuda').

        Raises:
            ValueError: If the source type is not 'gaussian' or 'gamma'.
        """
        self.device = device
        src_type = config["source"]["type"].lower()

        if src_type == "gaussian":
            self.mean = config["source"]["mean"]
            self.std = config["source"]["std"]
            self.sample_fn = lambda shape: torch.normal(
                self.mean, self.std, size=shape, device=device
            )
        elif src_type == "gamma":
            shape = config["source"]["shape"]  # also known as alpha or concentration
            scale = config["source"]["scale"]  # also known as theta, equal to 1/rate
            self.dist = torch.distributions.Gamma(concentration=shape, rate=1.0 / scale)
            self.sample_fn = lambda shape: self.dist.sample(shape).to(device)
        else:
            raise ValueError(f"Unsupported source type: {src_type}")

    def sample(self, shape: tuple) -> torch.Tensor:
        """Returns a sample from the distribution.

        Args:
            shape: The shape of the sample to draw.

        Returns:
            Tensor of shape `shape` sampled from the distribution.
        """
        return self.sample_fn(shape)
