import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

def skew_structure_loss(model, q, dq, DIM):

    q.requires_grad_(True)

    M = model.M(q)
    C = model.C(q,dq)

    M_dot = torch.zeros_like(M)

    for i in range(DIM):
        dM_dqi = autograd.grad(
            M[:,:,i].sum(),
            q,
            create_graph=True
        )[0]
        M_dot[:,:,i] = dM_dqi

    S = M_dot - 2*C

    return torch.mean((S + S.transpose(1,2))**2)