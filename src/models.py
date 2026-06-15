import torch
from torch import nn

class SimpleLinearRegressionModel(nn.Module):
    """
    Baseline single-input, single-output model for linear regression.
    """
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
    """
    Model for binary classification with included nonlinear activations.
    """
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
    """
    Baseline model for multiclass classification.
    """
    def __init__(self, input_features: int, output_features: int, hidden_units: int =8):
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
    """
    Model for multiclass classification with included nonlinearities.
    """
    def __init__(self, input_features: int, output_features: int, hidden_units: int = 8):
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

class ImageClassLinear(nn.Module):
    """
    Baseline model for FashionMNIST dataset classification
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

class ImageClassNonlinear(nn.Module):
    """
    FashionMNIST dataset classification model with nonlinearities.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

class ImageClassCNN(nn.Module):
    """
    CNN FashionMNIST dataset classification model based on TinyVGG architecture.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int,
                 img_height: int, img_width: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Get size of flattened tensor after passing conv blocks 1 and 2
        with torch.inference_mode():
            dummy = torch.zeros(1, input_shape, img_height, img_width)
            dummy = self.conv_block_2(self.conv_block_1(dummy))
            flatten_size = dummy.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flatten_size,
                      out_features=output_shape)
        )

    def forward(self, x):
        """
        # Forward pass of the linear regression model.
        :param x: torch.Tensor
        """
        """
        # It can be useful to run layers separately with print statements on a dummy example
        # to manually check dimensions.
        x = self.conv_block_1(x)
        # print(f"Output shape of Conv Block 1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of Conv Block 2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of Classifier: {x.shape}")
        return x
        """
        # Otherwise it is better to run the sequentially to leverage operator overlap
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
