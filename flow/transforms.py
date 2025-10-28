"""Data transformation classes and functions to fit, save, and load them."""

import json
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Mapping

import xarray as xr
import numpy as np


class Transform(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def fit(self, data: xr.DataArray) -> None:
        """Compute any statistics needed to parameterise the transform.

        Args:
            data: Data array to fit the transform on; typically the training data for a variable.
        """

    @abstractmethod
    def transform(self, data: xr.DataArray) -> xr.DataArray: ...

    @abstractmethod
    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a dictionary capturing the state of the transform.

        Returns:
            A dictionary containing all parameters needed to re-instantiate the transform,
            including any statistics computed during `fit`.
        """

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]):
        """Load a previously saved transform state.

        Args:
            state: A mapping containing all parameters needed to re-instantiate the transform.
        """


class IdentityTransform(Transform):
    def fit(self, data: xr.DataArray) -> None:
        pass

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        return data

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        return data

    def state_dict(self) -> dict[str, Any]:
        return {"name": "identity"}

    def load_state_dict(self, state: Mapping[str, Any]):
        pass


class Log1pTransform(Transform):
    def fit(self, data: xr.DataArray) -> None:
        pass

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        if (data < -1).any():
            raise ValueError("Log1pTransform input contains NaNs or values < -1.")
        return np.log1p(data)

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        return np.expm1(data)

    def state_dict(self) -> dict[str, Any]:
        return {"name": "log1p"}

    def load_state_dict(self, state: Mapping[str, Any]):
        pass


class SqrtTransform(Transform):
    def fit(self, data: xr.DataArray) -> None:
        pass

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        if (data < 0).any():
            raise ValueError("SqrtTransform input contains NaNs or negative values.")
        return np.sqrt(data)

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        return data**2

    def state_dict(self) -> dict[str, Any]:
        return {"name": "sqrt"}

    def load_state_dict(self, state: Mapping[str, Any]):
        pass


class NormalisationTransform(Transform):
    """Z-score normalisation transform."""

    def __init__(self, dims: list[str], **kwargs):
        self.dims = dims  # Dimensions to compute mean and std over
        self.mean = None
        self.std = None

    def fit(self, data: xr.DataArray) -> None:
        """Compute mean and std over specified dimensions, ignoring NaNs."""
        self.mean = float(data.mean(dim=self.dims, skipna=True).compute())
        self.std = max(
            float(data.std(dim=self.dims, skipna=True).compute()), 1e-6
        )  # Avoid division by zero

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        return data * self.std + self.mean

    def state_dict(self) -> dict:
        return {"name": "normalisation", "dims": self.dims, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict):
        self.dims = state["dims"]
        self.mean = state["mean"]
        self.std = state["std"]


class ClipOutputsTransform(Transform):
    """Clip model outputs to a specified min and max value.

    Note that this transform is a no-op during the `transform` step, and is only applied during
    `inverse_transform`. It is useful when the model outputs may be outside a valid range, e.g.
    negative precipitation values after reversing a normalisation transform (when using a model
    trained in normalised space).

    In most cases, this transform should be placed at the *start* of the list of transforms for
    an output variable, so that it is the last transform applied during inverse_transform.
    """

    def __init__(self, min: float | None = None, max: float | None = None, **kwargs):
        """Initialise.

        Args:
            min: Minimum value to clip to. If None, no minimum clipping is applied.
            max: Maximum value to clip to. If None, no maximum clipping is applied.

        Raises:
            ValueError: If min is not less than max.
        """
        self.min = min
        self.max = max

        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError(f"ClipOutputsTransform min {self.min} must be < max {self.max}.")

    def fit(self, data: xr.DataArray) -> None:
        pass

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        # No-op during transform. This transform is to clip model outputs only.
        return data

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        return np.clip(data, self.min, self.max)

    def state_dict(self) -> dict:
        return {"name": "clip_outputs", "min": self.min, "max": self.max}

    def load_state_dict(self, state: dict):
        self.min = state["min"]
        self.max = state["max"]


class ComposeTransform(Transform):
    """Compose several transforms together."""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def fit(self, data: xr.DataArray) -> None:
        for t in self.transforms:
            t.fit(data)
            data = t.transform(data)

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        for t in self.transforms:
            data = t.transform(data)
        return data

    def inverse_transform(self, data: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
        for t in reversed(self.transforms):
            data = t.inverse_transform(data)
        return data

    def state_dict(self) -> dict:
        return {
            "name": "compose",
            "transforms": [transform.state_dict() for transform in self.transforms],
        }

    def load_state_dict(self, state: dict):
        for transform, transform_state in zip(self.transforms, state["transforms"]):
            transform.load_state_dict(transform_state)


TRANSFORM_REGISTRY = {
    "identity": IdentityTransform,
    "log1p": Log1pTransform,
    "sqrt": SqrtTransform,
    "normalisation": NormalisationTransform,
    "clip_outputs": ClipOutputsTransform,
}
"""Mapping from transform names to classes."""


def fit_transforms(config: dict, ds: xr.Dataset) -> dict[str, dict[str, Transform]]:
    """Fit transforms using configuration and dataset.

    Args:
        config: Configuration dictionary.
        ds: Dataset containing variables to fit transforms on.

    Returns:
        transform_map: Dictionary mapping from type (predictors, targets, or source) and variable
            name to a fitted Transform instance.

    Raises:
        ValueError: If a variable specified in the config is not found in the dataset.
        ValueError: If a transform specification is invalid (e.g., missing 'name' key).
    """
    transform_map: dict[str, dict[str, Transform]] = {}

    groups = ["predictors", "targets"]
    if config["source"]["type"] == "coupled":
        groups.append("source")

    for group in groups:
        group_dict = config[group] if group != "source" else config["source"]["variables"]
        transform_map[group] = {}

        for variable_name, variable_config in group_dict.items():
            if variable_name not in ds:
                raise ValueError(
                    f"Variable {variable_name} not in dataset. Available variables: {list(ds.data_vars)}"
                )

            transform_specs = variable_config["transforms"]
            transforms: list[Transform] = []

            for spec in transform_specs:
                if not isinstance(spec, dict) or "name" not in spec:
                    raise ValueError(f"Invalid transform specification: {spec}")

                transform = TRANSFORM_REGISTRY[spec["name"]](**spec)
                transforms.append(transform)

            composed = ComposeTransform(transforms)
            composed.fit(ds[variable_name])
            transform_map[group][variable_name] = composed

    return transform_map


def save_transforms(transform_map: dict[str, dict[str, Transform]], path: str) -> None:
    """Save transforms to a JSON file.

    Args:
        transform_map: Dictionary mapping from variable name to a fitted Transform instance.
        path: Full path to save the transforms JSON file, including the filename.
    """
    serializable_dict = {}
    for group, group_transforms in transform_map.items():
        serializable_dict[group] = {
            var_name: transform.state_dict() for var_name, transform in group_transforms.items()
        }

    with open(path, "w") as f:
        json.dump(serializable_dict, f, indent=2)


def _load_transform_from_state(state: dict) -> Transform:
    """Instantiate a transform from a state dictionary

    Args:
        state: Dictionary which captures state required to instantiate a Transformation,
            including any previously-computed statistics, e.g,
            ```
            {
                "name": "normalisation,
                "dims": ["time", "grid_latitude", "grid_longitude"],
                "mean: 0.74,
                "std": 0.93
            }
            ```

    Returns:
        Transform instance with loaded state.

    Raises:
        ValueError: If the transform name is not "compose" or not in the TRANSFORM_REGISTRY.
    """
    name = state["name"]

    if name == "compose":
        transforms = []
        for transform_state in state["transforms"]:
            transforms.append(_load_transform_from_state(transform_state))
        return ComposeTransform(transforms)

    elif name in TRANSFORM_REGISTRY:
        transform = TRANSFORM_REGISTRY[name](**state)
        transform.load_state_dict(state)
        return transform

    else:
        raise ValueError(f"Unknown transform: {name}")


def load_transforms(path: str) -> dict[str, Transform]:
    """Load transforms from a JSON file.

    Args:
        path: Full path to the JSON file containing the transforms.

    Returns:
        transform_map: Dictionary mapping from variable name to a Transform instance with loaded
            state.
    """
    with open(path) as fh:
        states = json.load(fh)

    transform_map: dict[str, dict[str, Transform]] = {}
    for group, group_states in states.items():
        transform_map[group] = {
            var_name: _load_transform_from_state(state) for var_name, state in group_states.items()
        }

    return transform_map
