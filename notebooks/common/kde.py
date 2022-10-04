"""Implementation of Kernel Density Estimation (KDE) [1].
Kernel density estimation is a nonparameteric density estimation method. It works by
placing kernels K on each point in a "training" dataset D. Then, for a test point x, 
p(x) is estimated as p(x) = 1 / |D| \\sum_{x_i \\in D} K(u(x, x_i)), where u is some 
function of x, x_i. In order for p(x) to be a valid probability distribution, the kernel
K must also be a valid probability distribution.
References (used throughout the file):
    [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation


"""

"""
https://github.com/EugenHotaj/pytorch-generative
"""

from typing import List, Optional, Union

import numpy as np

import torch


# class GenerativeModel(abc.ABC, nn.Module):
#     """Base class inherited by all generative models in pytorch-generative.
#     Provides:
#         * An abstract `sample()` method which is implemented by subclasses that support
#           generating samples.
#         * Variables `self._c, self._h, self._w` which store the shape of the (first)
#           image Tensor the model was trained with. Note that `forward()` must have been
#           called at least once and the input must be an image for these variables to be
#           available.
#         * A `device` property which returns the device of the model's parameters.
#     """

#     def __call__(self, *args, **kwargs):
#         if getattr(self, "_c", None) is None and len(args[0].shape) == 4:
#             _, self._c, self._h, self._w = args[0].shape
#         return super().__call__(*args, **kwargs)

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     @abc.abstractmethod
#     def sample(self, n_samples):
#         ...


# class Kernel(abc.ABC, nn.Module):
#     """Base class which defines the interface for all kernels."""

#     def __init__(self, bandwidth=1.0):
#         """Initializes a new Kernel.
#         Args:
#             bandwidth: The kernel's (band)width.
#         """
#         super().__init__()
#         self.bandwidth = bandwidth

#     def _diffs(self, test_Xs: torch.Tensor, train_Xs: torch.Tensor):
#         """Computes difference between each x in test_Xs with all train_Xs."""
#         test_Xs = test_Xs.unsqueeze(1)
#         train_Xs = train_Xs.unsqueeze(0)
#         # test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
#         # train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
#         return test_Xs - train_Xs

#     @abc.abstractmethod
#     def forward(self, test_Xs, train_Xs):
#         """Computes log p(x) for each x in test_Xs given train_Xs."""

#     @abc.abstractmethod
#     def sample(self, train_Xs):
#         """Generates samples from the kernel distribution."""


# class ParzenWindowKernel(Kernel):
#     """Implementation of the Parzen window kernel."""

#     def forward(self, test_Xs, train_Xs):
#         abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
#         dims = tuple(range(len(abs_diffs.shape))[2:])
#         dim = np.prod(abs_diffs.shape[2:])
#         inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
#         coef = 1 / self.bandwidth**dim
#         return torch.log((coef * inside).mean(dim=1))

#     def sample(self, train_Xs):
#         device = train_Xs.device
#         noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
#         return train_Xs + noise


class GaussianKernel:
    def __init__(self, bandwidth: float):
        self.bandwidth = torch.tensor(bandwidth)

    @torch.no_grad()
    def scores(self, test_data: torch.Tensor, train_data: torch.Tensor) -> torch.Tensor:
        # TODO(eugenhotaj): This method consumes O(train_data * test_data) memory.
        # Implement an iterative version instead.

        # Computes difference between each x in test_data with all train_data.
        x = test_data.unsqueeze(1)
        y = train_data.unsqueeze(0)
        diffs = x - y / self.bandwidth

        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2  # type: ignore
        z = self._get_z(train_data)

        return torch.logsumexp(log_exp - z, dim=-1)

    def _get_z(self, train_data: torch.Tensor):
        n, d = train_data.shape
        n = torch.tensor(n, dtype=torch.float32)
        pi = torch.tensor(np.pi)

        z = 0.5 * d * torch.log(2 * pi) + d * torch.log(self.bandwidth) + torch.log(n)
        return z.to(train_data)

    # def sample(self, train_data: torch.Tensor) -> torch.Tensor:
    #     noise = torch.randn_like(train_data) * self.bandwidth
    #     return train_data + noise


class KernelDensity:
    """The KernelDensityEstimator model."""

    def __init__(
        self,
        train_data: Union[torch.Tensor, List[torch.Tensor]],
        bandwidth: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_data: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_data.
            default the batdwith is the silverman's rule of thunb
        """
        super().__init__()
        if isinstance(train_data, list):
            self.train_data = torch.concat(train_data)
        else:
            self.train_data = train_data
        assert len(self.train_data.shape) == 2, "Input cannot have more than two axes."
        self.train_data.requires_grad = False

        n, d = self.train_data.size()
        h = bandwidth or (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

        self.kernel = GaussianKernel(bandwidth=h)
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        y = self.train_data

        # Load on GPU
        x = x.to(self.device)
        y = y.to(self.device)

        # Compute
        score_samples = self.kernel.scores(x, y)

        # Back to cpu
        x = x.cpu()
        y = y.cpu()
        score_samples = score_samples.cpu()

        return score_samples

    # def sample(self, n_samples):
    #     idx = np.random.choice(range(len(self.train_data)), size=n_samples)
    #     return self.kernel.sample(self.train_data[idx])
