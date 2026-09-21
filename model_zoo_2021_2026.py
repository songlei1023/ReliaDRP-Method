import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 128
FAMILIES = {}


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GRL.apply(x, lam)


def split_omics(z):
    return [z[:, :CH], z[:, CH:2 * CH], z[:, 2 * CH:]]


class DenseGAT(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, n_nodes=3):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.adj_bias = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def forward(self, x):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, D // self.h).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, D // self.h).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (D // self.h) ** 0.5
        att = att + self.adj_bias.view(1, 1, L, L)
        att = self.drop(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class CRDNN(nn.Module):
    """Similarity-regularized deep net (2021)."""

    def __init__(self, cell_dim=384, drug_dim=128, hidden=128, dropout=0.3, w_sim=0.1):
        super().__init__()
        self.cell_enc = nn.Sequential(nn.Linear(cell_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.w_sim = w_sim

    def encode(self, cell, drug):
        return self.cell_enc(cell), self.drug_enc(drug)

    def forward(self, cell, drug):
        cz, dz = self.encode(cell, drug)
        return self.head(torch.cat([cz, dz], -1)).squeeze(-1)

    def _sim(self, x, z):
        xs = F.normalize(x, dim=-1)
        zs = F.normalize(z, dim=-1)
        n = x.shape[0]
        m = ~torch.eye(n, dtype=torch.bool, device=x.device)
        return F.mse_loss((xs @ xs.t())[m], (zs @ zs.t())[m])

    def compute_loss(self, cell, drug, y, ctx=None, w=None):
        cz, dz = self.encode(cell, drug)
        pred = self.head(torch.cat([cz, dz], -1)).squeeze(-1)
        loss = F.mse_loss(pred, y, reduction="none")
        loss = (loss * w).sum() / w.sum() if w is not None else loss.mean()
        return loss + self.w_sim * (self._sim(cell, cz) + self._sim(drug, dz))


class VAEN(nn.Module):
    """Variational autoencoder predictor (2021)."""

    def __init__(self, cell_dim=384, drug_dim=128, hidden=128, latent=64,
                 dropout=0.3, beta=1e-3, gamma=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(cell_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, cell_dim))
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(latent + hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.beta, self.gamma = beta, gamma

    def _z(self, cell):
        h = self.enc(cell)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8.0, 8.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std), mu, logvar
        return mu, mu, logvar

    def forward(self, cell, drug):
        z, _, _ = self._z(cell)
        d = self.drug_enc(drug)
        return self.head(torch.cat([z, d], -1)).squeeze(-1)

    def compute_loss(self, cell, drug, y, ctx=None, w=None):
        z, mu, logvar = self._z(cell)
        d = self.drug_enc(drug)
        pred = self.head(torch.cat([z, d], -1)).squeeze(-1)
        mse = F.mse_loss(pred, y, reduction="none")
        mse = (mse * w).sum() / w.sum() if w is not None else mse.mean()
        kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        rec = F.mse_loss(self.dec(z), cell)
        return mse + self.beta * kl + self.gamma * rec


class TGSA(nn.Module):
    """Topology-guided self-attention over omics tokens + drug token (2022)."""

    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128,
                 heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.rel_bias = nn.Parameter(torch.zeros(4, 4))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        om = [p(c) for p, c in zip(self.proj, split_omics(cell))]
        d = self.drug_proj(drug).unsqueeze(1)
        tok = torch.stack([*om, d.squeeze(1)], dim=1)
        out, _ = self.attn(tok, tok, tok, attn_mask=self.rel_bias)
        cellv = out[:, :3].mean(1)
        return self.head(torch.cat([cellv, out[:, 3]], -1)).squeeze(-1)


class CODEAE(nn.Module):
    """Context-aware deconfounding autoencoder (2022)."""

    def __init__(self, cell_dim=384, drug_dim=128, hidden=128, latent=64,
                 n_ctx=16, dropout=0.3, w_adv=0.5):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(cell_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, latent), nn.ReLU())
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(latent + hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.disc = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_ctx))
        self.w_adv = w_adv

    def forward(self, cell, drug):
        z = self.enc(cell)
        d = self.drug_enc(drug)
        return self.head(torch.cat([z, d], -1)).squeeze(-1)

    def compute_loss(self, cell, drug, y, ctx=None, w=None):
        z = self.enc(cell)
        d = self.drug_enc(drug)
        pred = self.head(torch.cat([z, d], -1)).squeeze(-1)
        mse = F.mse_loss(pred, y, reduction="none")
        mse = (mse * w).sum() / w.sum() if w is not None else mse.mean()
        if ctx is None:
            return mse
        logits = self.disc(grad_reverse(z, 1.0))
        return mse + self.w_adv * F.cross_entropy(logits, ctx)


class PanCancerDR(nn.Module):
    """Latent independent projection for domain generalization (2025)."""

    def __init__(self, cell_dim=384, drug_dim=128, hidden=128, latent=64,
                 dropout=0.3, w_indep=1.0):
        super().__init__()
        self.cell_enc = nn.Sequential(nn.Linear(cell_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, latent), nn.ReLU())
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, latent), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(latent * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.w_indep = w_indep

    def forward(self, cell, drug):
        zc, zd = self.cell_enc(cell), self.drug_enc(drug)
        return self.head(torch.cat([zc, zd], -1)).squeeze(-1)

    @staticmethod
    def _cross_cov(a, b):
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        cov = a.t() @ b / max(a.shape[0] - 1, 1)
        return (cov ** 2).mean()

    def compute_loss(self, cell, drug, y, ctx=None, w=None):
        zc, zd = self.cell_enc(cell), self.drug_enc(drug)
        pred = self.head(torch.cat([zc, zd], -1)).squeeze(-1)
        mse = F.mse_loss(pred, y, reduction="none")
        mse = (mse * w).sum() / w.sum() if w is not None else mse.mean()
        return mse + self.w_indep * self._cross_cov(zc, zd)


class MoGraphDRP(nn.Module):
    """Multi-omics graph + bilinear attention fusion (2025)."""

    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128,
                 heads=4, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.gat = DenseGAT(hidden, heads, dropout)
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.bilinear = nn.Bilinear(hidden, hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        tok = torch.stack([p(c) for p, c in zip(self.proj, split_omics(cell))], dim=1)
        tok = F.relu(self.gat(tok))
        cellv = tok.mean(1)
        d = F.relu(self.drug_proj(drug))
        return self.head(torch.cat([F.relu(self.bilinear(cellv, d)), cellv], -1)).squeeze(-1)


class MGATAF(nn.Module):
    """Multi-channel graph attention with adaptive fusion (2025)."""

    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128,
                 heads=4, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.gat = DenseGAT(hidden, heads, dropout)
        self.mod_w = nn.Parameter(torch.ones(len(omics_dims)) / len(omics_dims))
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        tok = torch.stack([p(c) for p, c in zip(self.proj, split_omics(cell))], dim=1)
        tok = F.relu(self.gat(tok))
        w = torch.softmax(self.mod_w, dim=0)
        cellv = (tok * w.view(1, -1, 1)).sum(1)
        d = F.relu(self.drug_proj(drug))
        return self.head(torch.cat([cellv, d], -1)).squeeze(-1)


NEW_MODELS = {
    "crdnn": CRDNN,
    "vaen": VAEN,
    "tgsa": TGSA,
    "codeae": CODEAE,
    "pancancer": PanCancerDR,
    "mograph": MoGraphDRP,
    "mgataf": MGATAF,
}

EXISTING_MODELS = {
    "bandrp": "BANDRP",
    "attn": "AttnOmics",
    "mmcl": "MMCL",
    "clclsa": "CLCLSA",
    "deepdtf": "DeepDTF",
    "delfos": "DELFOS",
    "fourierdrug": "FourierDrug",
    "mlp": "MultiMLP",
}


def build(name, **kw):
    if name in NEW_MODELS:
        return NEW_MODELS[name](**kw)
    exp = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "new_data", "exp"))
    if exp not in sys.path:
        sys.path.insert(0, exp)
    if name in EXISTING_MODELS:
        import models_survey as ms
        return getattr(ms, EXISTING_MODELS[name])()
    if name == "evi":
        from models_survey import MultiEviDRP
        return MultiEviDRP()
    if name == "evi2":
        from evidrp_v2 import EviDRPv2
        return EviDRPv2(interact=True, use_bn=True, attn=True)
    if name == "reliadrp":
        from reliadrp import ReliaDRP
        return ReliaDRP()
    raise ValueError(f"unknown model: {name}")


ALL_NAMES = list(NEW_MODELS) + list(EXISTING_MODELS) + ["evi", "evi2", "reliadrp"]
