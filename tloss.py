# Copyright 2023 University of Basel and Lucerne University of Applied Sciences and Arts Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


__author__ = "Alvaro Gonzalez-Jimenez"
__maintainer__ = "Alvaro Gonzalez-Jimenez"
__email__ = "alvaro.gonzalezjimenez@unibas.ch"
__license__ = "Apache License, Version 2.0"
__date__ = "2023-07-25"


import numpy as np
import torch
import torch.nn as nn

# NOTE: The mismatch between the paper and the code is because we offer a more general 
# formulation where Sigma is an arbitrary diagonal matrix. 
# For a detailed explanation, please refer to: https://github.com/Digital-Dermatology/t-loss/issues/2 

class TLoss(nn.Module):
    def __init__(
        self,
        config,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.config = config
        self.D = torch.tensor(
            (self.config.data.image_size * self.config.data.image_size),
            dtype=torch.float,
            device=config.device,
        )
 
        self.lambdas = torch.ones(
            (self.config.data.image_size, self.config.data.image_size),
            dtype=torch.float,
            device=config.device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=config.device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=config.device)
        self.reduction = reduction

    def forward(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
            first_term
            + second_term
            + third_term
            + fourth_term
            + fifth_term
            + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )
