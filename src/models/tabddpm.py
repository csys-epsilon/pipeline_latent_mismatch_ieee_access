"""
TabDDPM model and trainer.

Architecture and training schedule follow Manuscript v2.3 Section III-C:
    MLP with 512-512 hidden layers, 200 diffusion steps, 2000 epochs,
    Adam (lr=1e-4), β linear 1e-4 → 2e-2, batch_size 512.

Reference: Kotelnikov et al. (2023), "TabDDPM: Modelling Tabular Data with
Diffusion Models." (Original numerical preprocessing: QuantileTransformer.)
"""

import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):
    """MLP-based denoising network for tabular diffusion."""

    def __init__(self, x_dim, c_dim, cfg):
        super().__init__()
        d_layers = cfg.get('d_layers', [512, 512])
        dropout  = cfg.get('dropout', 0.1)
        t_dim    = cfg.get('time_emb_dim', 32)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim), nn.ReLU(), nn.Linear(t_dim, t_dim)
        )

        curr = x_dim + c_dim + t_dim
        layers = []
        for h in d_layers:
            layers += [nn.Linear(curr, h), nn.ReLU(), nn.Dropout(dropout)]
            curr = h
        layers.append(nn.Linear(curr, x_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, c, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        return self.model(torch.cat([x, c, t_emb], dim=1))


class TabDDPMTrainer:
    """
    Gaussian diffusion trainer for tabular data.

    Loss: weighted MSE between predicted and true noise. Samples drawn from the
    P97.5 heavy-tail region of each HK variable (encoded in `raw_mask`) receive
    a 2.5× reconstruction weight relative to non-tail samples. This ensures
    adequate gradient signal in the regime where cardiovascular events
    concentrate (Manuscript v2.3 Section III-C). The same weight is applied
    across every scaler condition; relative comparisons between scalers
    therefore remain unaffected by this choice.

    Pass `tail_weight=1.0` (or omit `raw_mask`) to recover the uniformly
    weighted DDPM objective.
    """

    TAIL_WEIGHT = 2.5    # P97.5 reconstruction weight; matches the run that
                         # produced the manuscript's reported numbers.

    def __init__(self, x_dim, c_dim, cfg, device):
        self.device = device
        self.cfg = cfg

        steps = cfg.get('diffusion_steps', 200)
        self.n_steps = steps[0] if isinstance(steps, list) else int(steps)

        self.model = MLPDiffusion(x_dim, c_dim, cfg).to(device)
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.get('lr', 1e-4),
            weight_decay=cfg.get('weight_decay', 1e-5),
        )

        self.beta = torch.linspace(
            cfg.get('beta_start', 1e-4),
            cfg.get('beta_end',   2e-2),
            self.n_steps,
        ).to(device)
        self.alpha_bar = torch.cumprod(1.0 - self.beta, dim=0)

        self.tail_weight = float(cfg.get('tail_weight', self.TAIL_WEIGHT))

    def train_step(self, x0, c, raw_mask=None):
        """
        One DDPM denoising step.

        Parameters
        ----------
        x0       : (B, D) clean preprocessed sample
        c        : (B, C) condition vector
        raw_mask : (B,) float tensor in {0, 1}; 1 indicates the sample is
                   in the P97.5 tail region of any HK variable. When omitted,
                   uniform weighting is applied.
        """
        self.model.train()
        bs = x0.size(0)
        t = torch.randint(0, self.n_steps, (bs,), device=self.device)
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        pred_noise = self.model(xt, c, t.float() / self.n_steps)

        sq = (pred_noise - noise) ** 2
        if raw_mask is None or self.tail_weight == 1.0:
            loss = sq.mean()
        else:
            # Per-sample weight: tail samples get `tail_weight`, others get 1.0
            w = torch.where(
                raw_mask.unsqueeze(-1) > 0,
                torch.tensor(self.tail_weight, device=self.device, dtype=sq.dtype),
                torch.tensor(1.0,              device=self.device, dtype=sq.dtype),
            )
            loss = (w * sq).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, c):
        """Reverse-time sampling from N(0, I) conditioned on c."""
        self.model.eval()
        n = c.size(0)
        x_dim = self.model.model[-1].out_features
        x = torch.randn((n, x_dim), device=self.device)

        for i in reversed(range(self.n_steps)):
            t = torch.full((n,), i, device=self.device).float() / self.n_steps
            alpha_i  = 1.0 - self.beta[i]
            beta_i   = self.beta[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha_i)) * (
                x - ((1.0 - alpha_i) / torch.sqrt(1.0 - self.alpha_bar[i]))
                * self.model(x, c, t)
            ) + torch.sqrt(beta_i) * noise

        return x
