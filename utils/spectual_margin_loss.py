import torch

def spectral_margin_loss(matrix, margin=0.05):

    eigvals = torch.linalg.eigvalsh(matrix)
    lambda_min = eigvals.min(dim=1).values

    return torch.mean(torch.relu(margin - lambda_min)**2)