import torch
import xarray as xr

from src.transforms import Transform


class UKCPDataset(torch.utils.data.Dataset):
    """Dataset for single-GCM/ensemble UKCP dataset."""

    def __init__(
        self,
        data_path: str,
        config: dict,
        transforms: dict[str, dict[str, Transform]],
    ):
        self.transforms = transforms
        self.config = config

        self.predictors = list(config["predictors"].keys())
        self.targets = list(config["targets"].keys())
        if config["source"]["type"] == "coupled":
            self.source = list(config["source"]["variables"].keys())
            if len(self.source) != len(self.targets):
                raise ValueError(
                    f"Number of coupled variables ({len(self.source)}) must equal the number "
                    f"target variables ({len(self.targets)})."
                )
        else:
            self.source = []

        self.variables = self.predictors + self.targets + self.source

        self.ds = xr.open_dataset(
            data_path,
            chunks={
                "time": 1,
                "grid_latitude": 64,
                "grid_longitude": 64,
            },
        )

        self.length = self.ds.dims["time"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.

        Transforms are applied.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            predictors: Tensor of conditioning variables, shape (no. conditioning variables, 64, 64)
            targets: Tensor of target variables, shape (no. targets, 64, 64)
            source: If config source type is "coupled", returns a tensor of coupled source
                variables, shape (no. coupled variables, 64, 64). Otherwise, returns an empty,
                placeholder tensor of the same shape as targets.
        """
        sample = self.ds[self.variables].isel(time=idx).load()

        predictors = self._get_group_tensor(sample, self.predictors, "predictors")
        targets = self._get_group_tensor(sample, self.targets, "targets")
        source = (
            self._get_group_tensor(sample, self.source, "source")
            if self.source
            else torch.empty_like(targets)
        )

        return predictors, targets, source

    def _get_group_tensor(self, sample: xr.Dataset, variables: list[str], name: str):
        """Get a tensor for a group of variables.

        Args:
            sample: Dataset sample containing the variables.
            variables: List of variable names to retrieve.
            name: Name of the group, one of "predictors", "targets" or "source".

        Returns:
            group_tensor: Tensor containing the transformed values of the specified variables.
        """
        tensors: list[torch.Tensor] = []

        for var in variables:
            transformed = torch.tensor(
                self.transforms[name][var].transform(sample[var]).values, dtype=torch.float32
            )
            tensors.append(transformed)

        group_tensor = torch.stack(tensors)

        return group_tensor
