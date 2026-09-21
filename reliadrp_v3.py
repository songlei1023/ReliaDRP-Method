"""ReliaDRP v3 — architecture upgrades for the four-layer sweep.

Round 1 (L3, single-cell expression):
  ReliaDRPExprV3
    * omics-token mixing: the PCA cell vector is split into k channel tokens and
      mixed by a self-attention block with a learnable adjacency bias (the
      in-domain mechanism that MoGraphDRP/MGATAF use), wrapped in residual +
      LayerNorm so it cannot hurt the plain-MLP path;
    * deeper residual trunk with SE-style channel gating;
    * optional BCE auxiliary loss (L3 labels are binary; pure NIG regression
      under-uses the classification signal);
    * identical output contract to ReliaDRPExpr (predict -> mu, sigma).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from reliadrp import nig_nll_elem, _block
from evidrp_v2 import softplus1, softplus_eps


class TokenMixer(nn.Module):
    """Self-attention over k channel tokens with learnable adjacency bias.

    Adopted from the DenseGAT design used by MoGraphDRP/MGATAF (2025), but
    wrapped as a pre-norm residual block so the model degrades gracefully to
    independent channel encoders when attention is not useful.
    """

    def __init__(self, d_tok, n_tok, heads=2, dropout=0.1):
        super().__init__()
        self.n_tok = n_tok
        self.norm = nn.LayerNorm(d_tok)
        self.q = nn.Linear(d_tok, d_tok)
        self.k = nn.Linear(d_tok, d_tok)
        self.v = nn.Linear(d_tok, d_tok)
        self.o = nn.Linear(d_tok, d_tok)
        self.h = heads
        self.adj = nn.Parameter(torch.zeros(n_tok, n_tok))
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_tok * 2),
                                nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_tok * 2, d_tok))

    def forward(self, tok):                       # tok: [B, k, D]
        B, K, D = tok.shape
        h = self.norm(tok)
        q = self.q(h).view(B, K, self.h, D // self.h).transpose(1, 2)
        k = self.k(h).view(B, K, self.h, D // self.h).transpose(1, 2)
        v = self.v(h).view(B, K, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (D // self.h) ** 0.5
        att = att + self.adj.view(1, 1, K, K)
        att = self.drop(torch.softmax(att, dim=-1))
        tok = tok + self.o((att @ v).transpose(1, 2).reshape(B, K, D))
        tok = tok + self.ff(self.norm(tok))
        return tok


class SEGate(nn.Module):
    """Squeeze-and-excitation gate over the hidden feature map (channel saliency)."""

    def __init__(self, d, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d // r), nn.ReLU(),
                                nn.Linear(d // r, d), nn.Sigmoid())

    def forward(self, z):
        return z * self.fc(z)


def _res(d, dropout):
    return nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d, d), nn.LayerNorm(d))


class ReliaDRPExprV3(nn.Module):
    """Single-cell expression model (L3) with token mixing + NIG head.

    Same external contract as ReliaDRPExpr:
      forward(x) -> (mu, v, a, b);  predict(x) -> (mu, sigma)
    compute_loss(x, y, ep, epochs, w_bce=...) — BCE auxiliary optional.
    """

    def __init__(self, d_in=384, hidden=256, n_tok=3, n_mix=2, dropout=0.2,
                 w_reg=0.05, w_bce=0.5, use_se=True):
        super().__init__()
        assert d_in % n_tok == 0, "d_in must split evenly into n_tok tokens"
        self.n_tok = n_tok
        self.d_tok = d_in // n_tok
        self.tok_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d_tok, hidden), nn.LayerNorm(hidden), nn.GELU())
            for _ in range(n_tok)])
        self.mixer = nn.ModuleList([TokenMixer(hidden, n_tok, dropout=dropout)
                                    for _ in range(n_mix)])
        trunk = [_block(hidden, hidden, dropout)]
        for _ in range(1):
            trunk.append(_res(hidden, dropout))
        self.trunk = nn.Sequential(*trunk)
        self.se = SEGate(hidden) if use_se else nn.Identity()
        self.out_mu = nn.Linear(hidden, 1)
        self.out_ev = nn.Linear(hidden, 3)
        self.w_reg = w_reg
        self.w_bce = w_bce

    def trunk_fwd(self, x):
        B = x.shape[0]
        d = self.d_tok
        tok = torch.stack([p(x[:, i * d:(i + 1) * d])
                           for i, p in enumerate(self.tok_proj)], dim=1)
        for m in self.mixer:
            tok = m(tok)
        z = tok.mean(1)
        z = self.trunk(z)
        return self.se(z)

    def forward(self, x):
        z = self.trunk_fwd(x)
        o = self.out_ev(z)
        return self.out_mu(z)[:, 0], softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, x, y, ep=0, epochs=30, mse_frac=0.5):
        z = self.trunk_fwd(x)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        v, a, b = softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])
        l_bce = F.binary_cross_entropy_with_logits(mu, y)
        if ep < mse_frac * epochs:
            return F.mse_loss(mu, y) + self.w_bce * l_bce
        nll = nig_nll_elem(y, mu, v, a, b).mean()
        reg = (torch.abs(y - mu) * (2.0 * v + a)).mean() * self.w_reg
        return nll + reg + self.w_bce * l_bce

    @torch.no_grad()
    def predict(self, x):
        mu, v, a, b = self(x)
        ale = b / (a - 1.0)
        epi = b / (v * (a - 1.0))
        return mu.cpu().numpy(), torch.sqrt(ale + epi).cpu().numpy()


class ReliaDRPComboV3(nn.Module):
    """Drug-combination model (L2) = ReliaDRPCombo + omics-token mixing.

    Cell 384-d vector is split into 3 x 128 channel tokens and mixed by the
    TokenMixer (MoGraphDRP's in-domain mechanism, residual-wrapped); the
    drug-pair encoder, bilinear term, NIG head and reliability head of
    ReliaDRPCombo are kept unchanged.
    """

    def __init__(self, d_cell=384, d_drug=128, hidden=128, dropout=0.1,
                 w_reg=0.05, w_rel=0.1, n_mix=1):
        super().__init__()
        assert d_cell % 3 == 0
        d_tok = d_cell // 3
        self.tok_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d_tok, hidden), nn.LayerNorm(hidden), nn.GELU())
            for _ in range(3)])
        self.mixer = nn.ModuleList([TokenMixer(hidden, 3, dropout=dropout)
                                    for _ in range(n_mix)])
        self.cell_out = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
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

    def cell_fwd(self, c):
        tok = torch.stack([p(c[:, i * (c.shape[1] // 3):(i + 1) * (c.shape[1] // 3)])
                           for i, p in enumerate(self.tok_proj)], dim=1)
        for m in self.mixer:
            tok = m(tok)
        return self.cell_out(tok.mean(1))

    def trunk(self, c, dr, dc):
        cv = self.cell_fwd(c)
        rv, cv2 = self.drug_enc(dr), self.drug_enc(dc)
        pair = self.pair_enc(torch.cat([rv, cv2, (rv - cv2).abs(), rv * cv2], -1))
        bil = self.bilinear(rv, cv2)
        return self.fusion(torch.cat([cv, pair, bil], -1))

    def forward(self, c, dr, dc):
        z = self.trunk(c, dr, dc)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, c, dr, dc, y, w=None, ep=0, epochs=12, mse_frac=0.5):
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


class ReliaDRPComboV4(nn.Module):
    """L2 v4: ablation-driven fixes on top of ReliaDRPCombo.

    * LayerNorm removed everywhere (plain ReLU blocks) — LN costs ~0.02
      Pearson on L2 (CSS regression has large-scale targets);
    * explicit cell x pair bilinear interaction (the MoGraphDRP mechanism the
      pair-only bilinear of v2 was missing);
    * feature dropout configurable (default 0.05, v2 used 0.1);
    * NIG + reliability heads and reliability-weighted loss unchanged.
    """

    def __init__(self, d_cell=384, d_drug=128, hidden=128, dropout=0.05,
                 w_reg=0.05, w_rel=0.1):
        super().__init__()
        plain = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Dropout(dropout),
                                           nn.Linear(o, o), nn.ReLU())
        self.cell_enc = plain(d_cell, hidden)
        self.drug_enc = plain(d_drug, hidden)
        self.pair_enc = plain(hidden * 4, hidden)
        self.bil_dd = nn.Bilinear(hidden, hidden, hidden)   # drug x drug
        self.bil_cd = nn.Bilinear(hidden, hidden, hidden)   # cell x pair
        self.fusion = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.out_mu = nn.Linear(hidden, 1)
        self.out_ev = nn.Linear(hidden, 3)
        self.rel_head = nn.Linear(hidden, 1)
        self.w_reg, self.w_rel = w_reg, w_rel

    def trunk(self, c, dr, dc):
        cv = self.cell_enc(c)
        rv, cv2 = self.drug_enc(dr), self.drug_enc(dc)
        pair = self.pair_enc(torch.cat([rv, cv2, (rv - cv2).abs(), rv * cv2], -1))
        bil_dd = self.bil_dd(rv, cv2)
        bil_cd = self.bil_cd(cv, pair)
        return self.fusion(torch.cat([cv, pair, bil_dd, bil_cd], -1))

    def forward(self, c, dr, dc):
        z = self.trunk(c, dr, dc)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, c, dr, dc, y, w=None, ep=0, epochs=12, mse_frac=0.5):
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


class ReliaDRPV3(nn.Module):
    """Single-drug model (L1/L4) = ReliaDRP + token mixing in the cell encoder.

    Identical heads, losses and I/O contract to ReliaDRP; only the cell encoder
    gains a residual TokenMixer over the three omics-channel tokens (and
    optional modality dropout for transfer robustness, off by default).
    """

    def __init__(self, cell_dim=384, drug_dim=128, d_hid=128, d_fus=128, dropout=0.1,
                 interact=True, use_bn=True, ch_layers=1, drug_layers=1,
                 w_reg=0.05, w_rel=0.1, n_ch=3, n_mix=1, mod_drop=0.0):
        super().__init__()
        from evidrp_v2 import ChannelCellEncoder
        self.interact = interact
        self.n_ch = n_ch
        self.mod_drop = mod_drop
        self.cell_enc = ChannelCellEncoder(d_hid, dropout, use_bn=use_bn,
                                           layers=ch_layers, n_ch=n_ch, attn=True)
        self.mixer = nn.ModuleList([TokenMixer(d_hid, n_ch, dropout=dropout)
                                    for _ in range(n_mix)])
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

    def cell_fwd(self, cell):
        ch = cell.shape[1] // self.n_ch
        hs = torch.stack([self.cell_enc.encs[i](cell[:, i * ch:(i + 1) * ch])
                          for i in range(self.n_ch)], dim=1)      # [B, n_ch, d_hid]
        for m in self.mixer:
            hs = m(hs)
        if self.cell_enc.attn:
            w = torch.softmax(self.cell_enc.attn_net(hs).squeeze(-1), dim=1)
            return (hs * w.unsqueeze(-1)).sum(1), w
        return hs.mean(1), None

    def trunk(self, cell, drug):
        if self.training and self.mod_drop > 0:
            keep = torch.rand(cell.shape[0], self.n_ch, 1, device=cell.device) > self.mod_drop
            cell = cell.view(cell.shape[0], self.n_ch, -1) * keep
            cell = cell.reshape(cell.shape[0], -1)
        cell_v, _ = self.cell_fwd(cell)
        drug_v = self.drug_enc(drug)
        z = torch.cat([cell_v, drug_v, cell_v * drug_v], -1) if self.interact \
            else torch.cat([cell_v, drug_v], -1)
        return self.fusion(z)

    def forward(self, cell, drug):
        z = self.trunk(cell, drug)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, softplus1(o[:, 0]), softplus1(o[:, 1]), softplus_eps(o[:, 2])

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=10, mse_frac=0.5):
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
