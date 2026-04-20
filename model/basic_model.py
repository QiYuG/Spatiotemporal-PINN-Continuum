import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# ==============================
# 通用MLP
# ==============================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# 下三角构造
# ==============================
def build_cholesky(raw, dim, EPS=1e-6):
    batch = raw.shape[0]
    L = torch.zeros(batch, dim, dim, device=raw.device)

    idx = 0
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                L[:,i,j] = torch.nn.functional.softplus(raw[:,idx]) + EPS
            else:
                L[:,i,j] = raw[:,idx]
            idx += 1
    return L


# ==============================
# 结构性凸势能网络
# V(q) = 1/2 q^T K(q) q
# K(q) = L(q)L(q)^T
# ==============================
class ConvexPotential(nn.Module):

    def __init__(self, DIM):
        super().__init__()
        self.DIM = DIM
        chol_dim = DIM*(DIM+1)//2
        self.K_net = MLP(DIM, chol_dim)

    def forward(self, q):

        raw = self.K_net(q)
        L = build_cholesky(raw, self.DIM)
        K = torch.bmm(L, L.transpose(1,2))

        q_vec = q.unsqueeze(-1)
        V = 0.5 * torch.bmm(
            q_vec.transpose(1,2),
            torch.bmm(K, q_vec)
        )

        return V.squeeze(-1)