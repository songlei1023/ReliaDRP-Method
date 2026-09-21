import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EXP = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "..", "new_data", "exp"))
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from evidrp_v2 import ChannelCellEncoder, softplus1, softplus_eps  # noqa: E402
from models import pred_evidential  # noqa: E402


def nig_nll_elem(t, mu, v, a, b):
    Omega = 2.0 * b * (1.0 + v)
    t1 = 0.5 * math.log(math.pi) - 0.5 * torch.log(v)
    t2 = -a * torch.log(Omega)
    t3 = (a + 0.5) * torch.log(v * (t - mu) ** 2 + Omega)
    t4 = torch.lgamma(a) - torch.lgamma(a + 0.5)
    return t1 + t2 + t3 + t4


class ReliaDRP(nn.Module):
    """Reliability-aware evidential DRP.

    Backbone: channel-attention multi-omics encoder + cell-drug interaction +
    NIG evidential head (same contract as EviDRP).
    New:
      * reliability head predicts the per-sample label reliability (aux task),
        shaping the representation with label-quality information;
      * reliability-weighted NIG NLL down-weights low-reliability labels.
    Calibration at inference: GC-CP (see run_e2_conformal_recovery.py).
    """

    def __init__(self, cell_dim=384, drug_dim=128, d_hid=128, d_fus=128, dropout=0.1,
                 interact=True, use_bn=True, ch_layers=1, drug_layers=1,
                 w_reg=0.05, w_rel=0.1, n_ch=3):
        super().__init__()
        self.interact = interact
        self.cell_enc = ChannelCellEncoder(d_hid, dropout, use_bn=use_bn,
                                           layers=ch_layers, n_ch=n_ch, attn=True)
        norm = (lambda d: nn.BatchNorm1d(d)) if use_bn else (lambda d: nn.Identity())
        deps = [nn.Linear(drug_dim, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(drug_layers - 1):
            deps += [nn.Linear(d_hid, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
        self.drug_enc = nn.Sequential(*deps)
        fus_in = d_hid * 3 if interact else d_hid * 2
        self.fusion = nn.Sequential(nn.Linear(fus_in, d_fus), nn.ReLU(), nn.Dropout(dropout))
        self.out_mu = nn.Linear(d_fus, 1)
        self.out_ev = nn.Linear(d_fus, 3)
        self.rel_head = nn.Linear(d_fus, 1)
        self.w_reg = w_reg
        self.w_rel = w_rel

    def trunk(self, cell, drug):
        cell_v, _ = self.cell_enc(cell)
        drug_v = self.drug_enc(drug)
        z = torch.cat([cell_v, drug_v, cell_v * drug_v], -1) if self.interact \
            else torch.cat([cell_v, drug_v], -1)
        return self.fusion(z)

    def forward(self, cell, drug):
        z = self.trunk(cell, drug)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=40,
                     mse_frac=0.5):
        z = self.trunk(cell, drug)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        v, a, b = softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])
        ww = torch.ones_like(mu) if w is None else w
        ww = ww / ww.mean().clamp(min=1e-8)

        rel_target = torch.ones_like(mu) if w is None else w
        l_rel = F.mse_loss(torch.sigmoid(self.rel_head(z))[:, 0], rel_target)

        if ep < mse_frac * epochs:
            l_point = (F.mse_loss(mu, y, reduction="none") * ww).mean()
            return l_point + self.w_rel * l_rel
        nll = (nig_nll_elem(y, mu, v, a, b) * ww).mean()
        reg = (torch.abs(y - mu) * (2.0 * v + a) * ww).mean() * self.w_reg
        return nll + reg + self.w_rel * l_rel

    @torch.no_grad()
    def predict(self, cell, drug):
        return pred_evidential(self, cell, drug)

    @torch.no_grad()
    def predict_reliability(self, cell, drug):
        return torch.sigmoid(self.rel_head(self.trunk(cell, drug)))[:, 0].cpu().numpy()


class ReliaDRPExpr(nn.Module):
    """ReliaDRP for single-cell / expression-only inputs (L3).

    No drug branch (each dataset is one treatment); an expression-native MLP encoder
    feeds a NIG evidential head. Trained with annealed MSE -> NIG NLL, so the model
    outputs a calibrated predictive interval for the (binary) response label.
    """

    def __init__(self, d_in=256, hidden=256, dropout=0.2, w_reg=0.05):
        super().__init__()
        self.enc = nn.Sequential(_block(d_in, hidden, dropout), _block(hidden, hidden, dropout))
        self.out_mu = nn.Linear(hidden, 1)
        self.out_ev = nn.Linear(hidden, 3)
        self.w_reg = w_reg

    def trunk(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.trunk(x)
        o = self.out_ev(z)
        return self.out_mu(z)[:, 0], softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, x, y, ep=0, epochs=40, mse_frac=0.5):
        z = self.trunk(x)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        v, a, b = softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])
        if ep < mse_frac * epochs:
            return F.mse_loss(mu, y)
        nll = nig_nll_elem(y, mu, v, a, b).mean()
        reg = (torch.abs(y - mu) * (2.0 * v + a)).mean() * self.w_reg
        return nll + reg

    @torch.no_grad()
    def predict(self, x):
        mu, v, a, b = self(x)
        ale = b / (a - 1.0)
        epi = b / (v * (a - 1.0))
        return mu.cpu().numpy(), torch.sqrt(ale + epi).cpu().numpy()


def _block(d_in, d_out, dropout):
    return nn.Sequential(
        nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(d_out, d_out), nn.LayerNorm(d_out), nn.ReLU())


class ReliaDRPCombo(nn.Module):
    """ReliaDRP for drug combinations (L2).

    Optimised v2 over the first version:
      * LayerNorm residual blocks in every encoder (training stability at higher capacity);
      * an explicit drug-pair interaction encoder over [h_row, h_col, |h_row-h_col|, h_row*h_col];
      * a cell-drug-pair bilinear term.
    A NIG evidential head and a reliability head are kept; training uses the
    reliability-weighted NIG NLL and inference is calibrated with GC-CP.
    """

    def __init__(self, d_cell=384, d_drug=128, hidden=128, dropout=0.1,
                 w_reg=0.05, w_rel=0.1):
        super().__init__()
        self.cell_enc = _block(d_cell, hidden, dropout)
        self.drug_enc = _block(d_drug, hidden, dropout)
        self.pair_enc = _block(hidden * 4, hidden, dropout)
        self.bilinear = nn.Bilinear(hidden, hidden, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
                                    nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.out_mu = nn.Linear(hidden, 1)
        self.out_ev = nn.Linear(hidden, 3)
        self.rel_head = nn.Linear(hidden, 1)
        self.w_reg, self.w_rel = w_reg, w_rel

    def trunk(self, c, dr, dc):
        cv = self.cell_enc(c)
        rv, cv2 = self.drug_enc(dr), self.drug_enc(dc)
        pair = self.pair_enc(torch.cat([rv, cv2, (rv - cv2).abs(), rv * cv2], -1))
        bil = self.bilinear(rv, cv2)
        return self.fusion(torch.cat([cv, pair, bil], -1))

    def forward(self, c, dr, dc):
        z = self.trunk(c, dr, dc)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, c, dr, dc, y, w=None, ep=0, epochs=40, mse_frac=0.5):
        z = self.trunk(c, dr, dc)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        v, a, b = softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])
        ww = torch.ones_like(mu) if w is None else (w / w.mean().clamp(min=1e-8))
        lock = F.mse_loss(torch.sigmoid(self.rel_head(z))[:, 0],
                          torch.ones_like(mu) if w is None else w)
        if ep < mse_frac * epochs:
            return (F.mse_loss(mu, y, reduction="none") * ww).mean() + self.w_rel * lock
        nll = (nig_nll_elem(y, mu, v, a, b) * ww).mean()
        reg = (torch.abs(y - mu) * (2.0 * v + a) * ww).mean() * self.w_reg
        return nll + reg + self.w_rel * lock
