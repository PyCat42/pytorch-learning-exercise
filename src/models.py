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

class CircleBinClassModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=30)
    self.layer_2 = nn.Linear(in_features=30, out_features=15)
    self.layer_3 = nn.Linear(in_features=15, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    """
    Forward pass of the linear regression model.
    :param x: torch.Tensor
    :return: torch.Tensor
    """
    # Apply ReLU after each hidden layer
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

class CircleMultiClassModel_Linear(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8):
    super().__init__()
    # Use nn.Sequential to stack NN layers
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=output_features)
    )
  def forward(self, x):
    return self.linear_layer_stack(x)

class CircleMultiClassModel_NonLinear(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features)
    )
  def forward(self, x):
    return self.linear_layer_stack(x)