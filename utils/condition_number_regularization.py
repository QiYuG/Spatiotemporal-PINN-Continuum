import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

def condition_number_regularization(matrix):

    matrix_inv = torch.linalg.inv(matrix)

    norm = torch.norm(matrix, dim=(1,2))
    norm_inv = torch.norm(matrix_inv, dim=(1,2))

    return torch.mean(norm * norm_inv)