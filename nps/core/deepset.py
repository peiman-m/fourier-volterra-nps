import torch
import torch.nn as nn

from ..utils.aggregate import Aggregator
from .mlp import MLP


class DeepSet(nn.Module):
    """
    Deep Set architecture for processing set-structured data.

    This model applies element-wise encoders followed by permutation-invariant aggregation.

    Args:
        aggregator: Aggregation module to pool encoded set elements.
        z_encoder: Optional encoder applied to aggregated features.
        xy_encoder: Optional encoder applied to concatenated input and query features.
        x_encoder: Optional encoder applied to input features.
        y_encoder: Optional encoder applied to query features.
    """

    def __init__(
        self,
        aggregator: Aggregator,
        z_encoder: MLP | None = None,
        xy_encoder: MLP | None = None,
        x_encoder: MLP | None = None,
        y_encoder: MLP | None = None,
    ):
        super().__init__()

        self.aggregator = aggregator
        self.z_encoder = z_encoder or nn.Identity()
        self.xy_encoder = xy_encoder or nn.Identity()
        self.x_encoder = x_encoder or nn.Identity()
        self.y_encoder = y_encoder or nn.Identity()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the Deep Set model.

        Args:
            x: Input features tensor.
            y: Query features tensor.
            mask: Optional mask for invalid elements.

        Returns:
            Aggregated representation of the set.
        """
        x_encoded = self.x_encoder(x)
        y_encoded = self.y_encoder(y)
        xy = torch.cat((x_encoded, y_encoded), dim=-1)
        z = self.xy_encoder(xy)
        z = self.aggregator(z, mask=mask)
        return self.z_encoder(z)
