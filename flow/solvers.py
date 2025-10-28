"""Numerical ODE solvers."""

from abc import ABC, abstractmethod

import torch


class Solver(ABC):
    def __init__(self, n_steps: int, alpha: float, model: torch.nn.Module, device: torch.device):
        """Initialise.

        Args:
            n_steps: Number of steps.
            alpha: Exponent for the time-stepping scheme. If alpha=1, then the time steps
                are uniformly spaced between 0 and 1 (constant step size). If alpha>1, then the
                step size is smaller near t=1 (the data/target side) and larger near t=0 (the
                noise/source side).
            model: Neural network model representing the velocity/vector field.
            device: Device
        """
        self.n_steps = n_steps
        self.alpha = alpha
        self.device = device
        self.model = model
        self.time_steps = (
            1 - (1 - torch.tensor(range(self.n_steps + 1)) / self.n_steps) ** self.alpha
        )  # shape (n_steps + 1, )
        self.step_sizes = self.time_steps[1:] - self.time_steps[:-1]  # shape (n_steps, )

    def simulate_ode(self, x_0: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Simulate the ODE.

        Args:
            x_0: Tensor of shape (batch_size, num_target_variables, height, width)
            conditioning: Conditioning variables,
                shape (batch_size, num_conditioning_variables, height, width)

        Returns:
            x_1: Tensor of shape (batch_size, num_target_variables, height, width)
        """
        x_current = x_0.to(self.device)

        for idx, step_size in enumerate(self.step_sizes):
            t_current = self.time_steps[idx].item()
            t_current = torch.full(
                (x_current.shape[0], 1), t_current, device=self.device
            )  # shape (batch_size, 1)
            x_current = self.step(x_current, t_current, conditioning, step_size.item())

        return x_current

    @abstractmethod
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        conditioning: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """Take a single step of the ODE solver.

        Args:
            x_current: Current state, shape (batch_size, num_target_variables, height, width)
            t_current: Current time between 0 and 1, shape (batch_size, 1)
            conditioning: Conditioning variables,
                shape (batch_size, num_conditioning_variables, height, width)
            step_size: Step size for this step.

        Returns:
            x_next: Next state, shape (batch_size, num_target_variables, height, width
        """
        pass


class EulerSolver(Solver):
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        conditioning: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        x_next = x_current + step_size * self.model(x_current, t_current, conditioning)
        return x_next


class HeunSolver(Solver):
    def step(
        self,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        conditioning: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        dx = self.model(x_current, t_current, conditioning)
        x_next_estimate = x_current + step_size * dx
        x_next = x_current + (step_size / 2) * (
            dx + self.model(x_next_estimate, t_current, conditioning)
        )
        return x_next
