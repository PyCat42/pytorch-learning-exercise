import torch
import seaborn as sns
import matplotlib.pyplot as plt

def get_accuracy(y_true, y_pred):
  """
  Calculates accuracy.
  Same as these built-in functions:
  - PyTorch
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)
    torch_accuracy = metric(y_pred, y_test)
  - Scikit Learn
    sklearn.metrics.accuracy_score(y_test.cpu(), y_pred.cpu())
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :return: accuracy = 100 * True Positive / (True Positive + True Negative)
  """
  correct = torch.eq(y_true.cpu(), y_pred.cpu()).sum().item()
  return (correct / len(y_pred)) * 100

def get_precision(y_true, y_pred, target_class):
  """
  Get precision for target class.
  Higher precision => fewer false positives.
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param target_class: target class label
  :return: precision = True Positive / (True Positive + False Positive)
  """
  true_positives = ((y_pred == target_class) & (y_true == target_class)).sum().item()
  false_positives = ((y_pred == target_class) & (y_true != target_class)).sum().item()
  return 100 * true_positives / (true_positives + false_positives + 1e-8)

def precision_macro(y_true, y_pred, num_classes):
  """
  Precision averaged over all classes.
  Same ase these built-in functions:
  - PyTorch:
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    torch_precision = metric(y_pred, y_test)
  - Scikit Learn
    sklearn.metrics.precision_score(y_test.cpu(), y_pred.cpu(), average="macro")
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param num_classes: number of classes
  :return: precision = sum of class precisions / number of classes
  """
  precision_vals = []

  for c in range(num_classes):
    precision_vals.append(get_precision(y_true, y_pred, c))

  return sum(precision_vals) / num_classes

def get_recall(y_true, y_pred, target_class):
  """
  Calculates recall for target class.
  Higher recall => fewer false negatives.
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param target_class: target class label
  :return: recall = True Positives / (True Positives + False Negatives)
  """
  true_positives = ((y_pred == target_class) & (y_true == target_class)).sum().item()
  false_negatives = ((y_pred != target_class) & (y_true == target_class)).sum().item()
  return 100 * true_positives / (true_positives + false_negatives + 1e-8)

def recall_macro(y_true, y_pred, num_classes):
  """
  Recall averaged over all classes.
  Same ase these built-in functions:
  - PyTorch:
    metric = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    torch_recall = metric(y_pred, y_test)
  - Scikit Learn
    sklearn.metrics.recall_score(y_test.cpu(), y_pred.cpu(), average="macro")
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param num_classes: number of classes
  :return: recall = sum of class recalls / number of classes
  """
  recall_vals = []

  for c in range(num_classes):
    recall_vals.append(get_recall(y_true, y_pred, c))

  return sum(recall_vals) / num_classes

def get_f1_score(y_true, y_pred, target_class):
  """
  Calculates F1 score for target class.
  Good overall metrics for classification.
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param target_class: target class label
  :return: f1 score = 2 * precision * recall / (precision + recall)
  """
  precision = get_precision(y_true, y_pred, target_class)
  recall = get_recall(y_true, y_pred, target_class)

  return 2 * precision * recall / (precision + recall + 1e-8)

def f1_score_macro(y_true, y_pred, num_classes):
  """
  F1 score averaged over all classes.
  Same ase these built-in functions:
  - PyTorch:
    metric = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    torch_f1 = metric(y_pred, y_test)
  - Scikit Learn
    sklearn.metrics.f1_score(y_test.cpu(), y_pred.cpu(), average="macro")
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param num_classes: number of classes
  :return: f1 score = sum of class f1 scores / number of classes
  """
  f1_score_vals = []

  for c in range(num_classes):
    f1_score_vals.append(get_f1_score(y_true, y_pred, c))

  return sum(f1_score_vals) / num_classes

def get_confusion_matrix(y_true, y_pred, num_classes, normalize=None, cmap="viridis"):
  """
  Calculate confusion matrix.
  Multiple normalization choices implemented.
  Plot calculated matrix using seaborn's heatmap.
  Same as these built-in functions:
  - PyTorch:
    metric = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)
    torch_confusion = metric(y_pred, y_test, normalize=...)
    fig, ax = torch_confusion_metric.plot()
    plt.show()
  - Scikit Learn:
    sk_confusion = sklearn.metrics.confusion_matrix(y_test.cpu(), y_pred.cpu())
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=sk_confusion, display_labels=[...])
    disp.plot(cmap=cmap)
    plt.show()
  :param y_true: true labels
  :param y_pred: labels predicted by the model
  :param num_classes: number of classes
  :param normalize: None (default) - not normalized
                    r - normalized by rows
                    c - normalized by columns
                    a - general normalization
  :return:
  """
  matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

  for t, p in zip(y_true, y_pred):
    matrix[t, p] += 1

  if normalize is None:
    print("Not normalized")
  elif normalize == "r":
    print("Normalized by row:")
    row_sums = matrix.sum(dim=1, keepdim=True)
    matrix /= row_sums.clamp(min=1)
  elif normalize == "c":
    print("Normalized by column:")
    column_sums = matrix.sum(dim=0, keepdim=True)
    matrix /= column_sums.clamp(min=1)
  elif normalize == "a":
    print("Global normalization:")
    matrix /= matrix.sum().clamp(min=1)
  else:
    print("Error: Normalization option does not exist! Confusion matrix not normalized!")

  plt.figure(figsize=(6, 5))
  sns.heatmap(matrix, annot=True, cmap=cmap)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()

  return matrix
