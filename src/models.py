import torch
from torch import nn

class SimpleLinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1)

  def forward(self, x):
    """
    Forward pass of the linear regression model.
    :param x: torch.Tensor
    :return: torch.Tensor
    """
    return self.linear_layer(x)

