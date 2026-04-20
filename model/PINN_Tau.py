import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from model.basic_model import MLP, build_cholesky, ConvexPotential


# ==============================
# 4DOF PINN Tau
# ==============================
class PINN_Tau(nn.Module):

    def __init__(self, DIM, device):
        super().__init__()
        self.DIM = DIM
        self.device = device

        chol_dim = DIM*(DIM+1)//2

        self.mass_net = MLP(DIM, chol_dim)
        self.damp_net = MLP(DIM*2, chol_dim)

        self.potential_net = ConvexPotential(DIM)

    def M(self, q):
        raw = self.mass_net(q)
        L = build_cholesky(raw, self.DIM)
        return torch.bmm(L, L.transpose(1,2))

    def D(self, q, dq):
        x = torch.cat([q,dq], dim=1)
        raw = self.damp_net(x)
        L = build_cholesky(raw, self.DIM)
        return torch.bmm(L, L.transpose(1,2))

    def C(self, q, dq):

        q.requires_grad_(True)
        M = self.M(q)

        batch = q.shape[0]
        C = torch.zeros(batch, self.DIM, self.DIM, device=self.device)

        for k in range(self.DIM):
            for j in range(self.DIM):
                for i in range(self.DIM):

                    dM_ik_dqj = autograd.grad(
                        M[:,i,k].sum(), q,
                        create_graph=True
                    )[0][:,j]

                    dM_ij_dqk = autograd.grad(
                        M[:,i,j].sum(), q,
                        create_graph=True
                    )[0][:,k]

                    dM_jk_dqi = autograd.grad(
                        M[:,j,k].sum(), q,
                        create_graph=True
                    )[0][:,i]

                    C[:,i,j] += 0.5*(dM_ik_dqj + dM_ij_dqk - dM_jk_dqi)*dq[:,k]

        return C

    def forward(self, q, dq, ddq):

        q.requires_grad_(True)

        M = self.M(q)
        C = self.C(q,dq)
        D = self.D(q,dq)

        V = self.potential_net(q)
        gradV = autograd.grad(V.sum(), q, create_graph=True)[0]

        tau_pred = (
            torch.bmm(M, ddq.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1)
            + gradV
        )

        return tau_pred