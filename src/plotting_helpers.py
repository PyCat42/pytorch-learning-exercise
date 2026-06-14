import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions(train_data, train_labels,
                     test_data, test_labels,
                     predictions=None,
                     filename="predictions.png"):
  """
  Plots training data, test data and compares predictions (if they exist).
  :param train_data:
  :param train_labels:
  :param test_data:
  :param test_labels:
  :param predictions:
  :param filename:
  :return:
  """

  plt.figure(figsize=(8, 6))

  # Plot training data
  plt.scatter(train_data, train_labels, c="blue", s=4, label="Training data")

  # Plot test data
  plt.scatter(test_data, test_labels, c="orange", s=4, label="Test data")

  # Plot predictions if they exist
  if predictions is not None:
    plt.scatter(test_data, predictions, c="red", s=4, label="Predictions")

  plt.legend()

  if filename is not None:
    plt.savefig(filename)

  plt.show()

def plot_curves(epoch_values, train_values, test_values, var_to_plot, filename=f"curves.png"):
  """
  Plots training and test loss curves.
  :param epoch_values: list of number of epochs
  :param train_values: list of training values
  :param test_values: list of testing values
  :param var_to_plot: "loss" or "accuracy"
  :param filename: name of the file to save the curves
  :return:
  """
  plt.figure(figsize=(8, 6))

  # Plot train loss curve
  plt.plot(epoch_values, torch.tensor(train_values).cpu().numpy(), label=f"Train {var_to_plot}")

  # Plot test loss curve
  plt.plot(epoch_values, torch.tensor(test_values).cpu().numpy(), label=f"Test {var_to_plot}")

  plt.title(f"{var_to_plot.capitalize()} curves")
  plt.xlabel("Epoch")
  plt.ylabel(var_to_plot.capitalize())
  plt.legend()

  if filename is not None:
    plt.savefig(filename)

  plt.show()

def plot_decision_boundary(model, X, y, cmap="cool"):
  """
  Plots the decision boundary fitted by model for binary or multiclass classification.
  :param model: fitted model
  :param X: data
  :param y: labels
  :return:
  """
  # Put everything on CPU
  model.to("cpu")
  X = X.to("cpu")
  y = y.to("cpu")

  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, 500),
                               np.linspace(y_min, y_max, 500))
  X_from_grid = torch.from_numpy(np.column_stack((x_mesh.ravel(), y_mesh.ravel()))).type(torch.float)

  model.eval()
  with torch.inference_mode():
    y_logits = model(X_from_grid)

  if len(torch.unique(y)) > 2:
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
  else:
    y_pred = torch.round(torch.sigmoid(y_logits))

  y_pred = y_pred.reshape(x_mesh.shape).detach().numpy()

  plt.contourf(x_mesh, y_mesh, y_pred, cmap=cmap, alpha=0.2)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=4, cmap=cmap)
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)

def plot_test_train_decision_boundary(model, X_train, y_train, X_test, y_test, filename=None):
  """
  Plots the decision boundary fitted by model for binary or multiclass classification
  for train and test data side by side.
  :param model: fitted model
  :param X: data
  :param y: labels
  :return:
  """
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.title("Train")
  plot_decision_boundary(model, X_train, y_train)

  plt.subplot(1, 2, 2)
  plt.title("Test")
  plot_decision_boundary(model, X_test, y_test)

  if filename is not None:
    plt.savefig(filename)

  plt.show()
