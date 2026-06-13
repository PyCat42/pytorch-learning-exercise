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
  plt.savefig(filename)
  plt.show()

def plot_loss_curves(epoch_count, train_loss_values, test_loss_values, filename="loss_curves.png"):
  """
  Plots training and test loss curves.
  :param epoch_count:
  :param train_loss_values:
  :param test_loss_values:
  :param filename:
  :return:
  """
  plt.figure(figsize=(8, 6))

  # Plot train loss curve
  plt.plot(epoch_count, torch.tensor(train_loss_values).cpu().numpy(), label="Train loss")

  # Plot test loss curve
  plt.plot(epoch_count, torch.tensor(test_loss_values).cpu().numpy(), label="Test loss")

  plt.title("Loss curves")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()

  plt.savefig(filename)
  plt.show()

def plot_decision_boundary(model, X, y, cmap=plt.cm.RdYlBu):
  # Put everything on CPU
  model.to("cpu")
  X = X.to("cpu")
  y = y.to("cpu")

  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  x_mesh, y_mesh = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
  X_from_grid = torch.from_numpy(np.column_stack((x_mesh.ravel(), y_mesh.ravel())))

  model.eval()
  with torch.inference_mode():
    y_logits = model(X_from_grid)

  if len(torch.unique(y)) > 2:
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
  else:
    y_pred = torch.round(torch.sigmoid(y_logits))

  y_pred = y_pred.reshape(x_mesh.shape).detach().numpy()

  plt.contourf(x_mesh, y_mesh, y_pred, cmap=plt.cm.RdYlBu, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=4, cmap=plt.cm.RdYlBu)
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
