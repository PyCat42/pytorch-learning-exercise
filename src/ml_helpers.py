import torch

def train_test_split(X_data, y_data, train_size=0.8):
    """
    Split data into train and test sets.
    Built-in function:
    sklearn.model_selection.train_test_split(X, y, test_size=None, train_size=None,
     random_state=None, shuffle=True, stratify=None)
    :param X_data: data
    :param y_data: labels
    :param train_size: size of training set (from 0 to 1)
    :return: X_train, y_train, X_test, y_test
    """
    num_samples = len(X_data)

    # Shuffle indices for randomness
    indices = torch.randperm(num_samples)

    # Calculate how much data belong to training sample
    train_split = int(train_size * len(X_data))

    train_indices = indices[:train_split]
    test_indices = indices[train_split:]

    X_train, y_train = X_data[train_indices], y_data[train_indices]
    X_test, y_test = X_data[test_indices], y_data[test_indices]

    return X_train, y_train, X_test, y_test

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_func,
               device: torch.device = "cpu",
               print_status: bool = True):
    """
    Functionalizing training loop.
    :param model: model to train
    :param data_loader: data loader containing training data
    :param loss_func: loss function
    :param optimizer: optimizer
    :param accuracy_func: accuracy function
    :param device: "cpu" or "gpu
    :param print_status: if True (default) printing loss and accuracy values during training
    :return:
    """
    # Initialize loss and accuracy to 0
    train_loss, train_acc = 0.0, 0.0

    # Put model on target device
    model.to(device)

    # Loop through the training batches
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Put data and label tensors to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate the loss for current batch
        loss = loss_func(y_pred, y)

        # Accumulate loss and accuracy for all batches
        train_loss += loss.item()
        train_acc += accuracy_func(y, y_pred.argmax(dim=1)).item()

        # Zero out optimizer gradient
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step the optimizer - optimizer updates once PER BATCH!
        optimizer.step()

    # Divide total train loss by length of train dataloader to get average loss per batch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    if print_status:
        # Print progress
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_func: torch.nn.Module,
              accuracy_func,
              device: torch.device = "cpu",
              print_status: bool = True):
    """
    Functionalizing testing loop.
    :param model: model to train
    :param data_loader: data loader containing training data
    :param loss_func: loss function
    :param optimizer: optimizer
    :param accuracy_func: accuracy function
    :param device: "cpu" or "gpu
    :param print_status: if True (default) printing loss and accuracy values during training
    :return:
    """
    # Initialize loss and accuracy to 0
    test_loss, test_acc = 0.0, 0.0

    # Put model in evaluation mode
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Put data and label tensors to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Accumulate loss and accuracy for all batches
            test_loss += loss_func(test_pred, y).item()
            test_acc += accuracy_func(y, test_pred.argmax(dim=1)).item()

        # Calculate test loss and accuracy average per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    if print_status:
        # Print progress
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")

def get_training_time(start_time: float, end_time: float, device: torch.device ="cpu"):
    """
    Return formatted training time.
    :param start_time:
    :param end_time:
    :param device:
    :return:
    """
    total_time = end_time - start_time
    print(f"Total experiment time on {device}: {total_time:.3f} seconds.")
    return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               accuracy_func: torch.nn.Module,
               device: torch.device = "cpu"):
    """
    Evaluate model performance.
    :param model: trained model
    :param data_loader: data loader containing testing data
    :param loss_func: loss function
    :param accuracy_func: accuracy function
    :param device: "cpu" or "gpu
    :return:
    """
    # Initialize loss and accuracy to 0
    loss, acc = 0.0, 0.0

    # Evaluate model
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Put tensors to device
            X, y = X.to(device), y.to(device)

            # Make predictions with trained model
            y_pred = model(X)

            # Accumulate loss and accuracy for all batches
            loss += loss_func(y_pred, y)
            acc += accuracy_func(y, y_pred.argmax(dim=1))

        # Get average loss and accuracy
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # works when model was created with the class!
            "model_loss": loss.item(),
            "model_acc": acc.item()}

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device):
    """
    Calculates probabilities that data sample belongs to
    each of the classes using trained model.
    :param model: trained model
    :param data: list of data to make predictions on
    :param device: "cpu" or "gpu
    :return:
    """
    pred_probs = []

    # Put model to device
    model.to(device)

    # Put model in eval mode
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Model expects data to have a batch dimension!
            # sample: [color_channel, height, width]
            # model: [batch_dim, color_channel, height, width]
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Make predictions
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # No argmax here!
            # We want to be able to see probabilities for each of the classes

            # Append to predictions list
            # matplotlib is going to need predictions on CPU!
            pred_probs.append(pred_prob.cpu())

    # Stack list to return a prediction tensor
    return torch.stack(pred_probs)
