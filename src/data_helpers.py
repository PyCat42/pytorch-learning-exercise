import torch

def dummy_data_lin_reg(weight, bias,
                       start, end, step):
    """
    Create simple linear regression dummy data
    :param weight:
    :param bias:
    :param start:
    :param end:
    :param step:
    :return: X, y
    """
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    return X, y