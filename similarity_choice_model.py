import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helper utilities
# -----------------------------

def row_normalize(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize a matrix to sum to 1 over the last dimension.
    Works with shape (..., N, N).
    """
    mat = torch.clamp(mat, min=0.0)  # ensure non-negative before normalization
    rowsum = mat.sum(dim=-1, keepdim=True).clamp_min(eps)
    return mat / rowsum


def topk_mask(sim: torch.Tensor, k: int, fill_value: float = -1e9, exclude_self: bool = True) -> torch.Tensor:
    """Create an additive mask that keeps only top-k per row in sim and masks others with fill_value.
    sim: (N, N) or (B, N, N)
    Returns mask with same shape to be added to attention logits.
    """
    if sim.dim() == 2:
        sim = sim.unsqueeze(0)
    B, N, _ = sim.shape

    device = sim.device
    arange_idx = torch.arange(N, device=device)

    # Optionally drop diagonal (self edges) before choosing top-k
    if exclude_self:
        sim = sim.clone()
        sim[..., arange_idx, arange_idx] = -float('inf')

    # Compute top-k indices per row
    topk_vals, topk_idx = torch.topk(sim, k=min(k, N - (1 if exclude_self else 0)), dim=-1)

    # Build mask: start with all masked, then unmask top-k
    mask = torch.full((B, N, N), fill_value, device=device)
    # Gather to create boolean selector
    gather_helper = torch.zeros_like(mask, dtype=torch.bool)
    gather_helper.scatter_(-1, topk_idx, True)
    mask = torch.where(gather_helper, torch.zeros_like(mask), mask)

    if exclude_self:
        mask[..., arange_idx, arange_idx] = fill_value

    return mask if mask.shape[0] > 1 else mask.squeeze(0)


# -----------------------------
# Monotone (non-increasing) price utility
# -----------------------------
class MonotonePriceNet(nn.Module):
    """A 1D monotone-decreasing function u(p).

    We parameterize u(p) = b - sum_m w_m * softplus(alpha_m * (p - c_m))
    with w_m = softplus(\tilde{w}_m) >= 0 and alpha_m = softplus(\tilde{a}_m) >= 0.
    Each term contributes a non-negative, increasing function in p; subtracting
    enforces du/dp <= 0 globally.
    """
    def __init__(self, n_terms: int = 8, init_centers: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_terms = n_terms
        self.w_raw = nn.Parameter(torch.zeros(n_terms))
        self.a_raw = nn.Parameter(torch.zeros(n_terms))
        if init_centers is None:
            # initialize centers between [0, 1] (prices should be scaled)
            centers = torch.linspace(0.0, 1.0, n_terms)
        else:
            centers = init_centers
        self.c = nn.Parameter(centers.clone())
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (...,) price (recommend scaling to ~[0,1]) -> (...,) scalar utility"""
        # reshape to (..., 1, n_terms) broadcasting
        p_exp = p.unsqueeze(-1)
        w = F.softplus(self.w_raw)  # (M,)
        a = F.softplus(self.a_raw)  # (M,)
        # softplus(alpha*(p-c)) is increasing in p
        bumps = F.softplus(a * (p_exp - self.c))  # (..., M)
        u = self.bias - (bumps * w).sum(dim=-1)
        return u


# -----------------------------
# Multi-Head Similarity-Augmented Attention (sparse)
# -----------------------------
class MultiHeadSimilarityAttention(nn.Module):
    def __init__(self, d_in: int, d_model: int = 64, n_heads: int = 4, topk: int = 16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.topk = topk

        self.q_proj = nn.Linear(d_in, d_model)
        self.k_proj = nn.Linear(d_in, d_model)
        self.v_proj = nn.Linear(d_in, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable mixing weight for similarity bias
        self.tau_raw = nn.Parameter(torch.tensor(0.0))  # tau = softplus(tau_raw) >= 0

    def forward(self, x: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_in) item representations (exclude price feature)
        sim: (N, N) or (B, N, N) non-negative similarity (will be row-normalized)
        returns: (B, N, d_model)
        """
        B, N, _ = x.shape
        device = x.device

        # Project to Q,K,V
        Q = self.q_proj(x)  # (B, N, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: (B, h, N, d_head)
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        Qh, Kh, Vh = map(split_heads, (Q, K, V))
        scale = 1.0 / math.sqrt(self.d_head)

        # Attention logits from content: (B, h, N, N)
        content_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale

        # Similarity bias (shared across heads). Build k-NN mask and row-normalize.
        if sim.dim() == 2:
            sim = sim.unsqueeze(0).expand(B, -1, -1).contiguous()
        sim = torch.clamp(sim, min=0.0)
        # keep only top-k neighbors per row
        knn_mask = topk_mask(sim, k=self.topk, fill_value=-1e9, exclude_self=True)
        sim_masked = sim + (knn_mask == 0) * 0.0  # ensure shape/grad friendly
        # convert mask with -1e9 into -inf addend
        add_mask = knn_mask
        sim_masked = torch.where(add_mask < 0, torch.zeros_like(sim), sim)
        sim_bias = row_normalize(sim_masked)  # (B, N, N)

        tau = F.softplus(self.tau_raw)
        # Broadcast sim_bias to heads
        sim_bias_h = sim_bias.unsqueeze(1).expand(B, self.n_heads, N, N)

        logits = content_logits + tau * sim_bias_h
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, Vh)  # (B, h, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)
        return out


# -----------------------------
# Main model: Similarity-Aware Choice with Outside Option & Constraints
# -----------------------------
class SimilarityAwareChoiceModelImproved(nn.Module):
    def __init__(
        self,
        n_features: int,
        price_index: int,
        d_model: int = 64,
        n_heads: int = 4,
        topk: int = 16,
        base_hidden: int = 64,
        interaction_hidden: int = 64,
        price_terms: int = 8,
        outside_hidden: int = 32,
        attn_residual_scale: float = 1.0,
    ):
        super().__init__()
        assert 0 <= price_index < n_features
        self.price_index = price_index
        self.n_features = n_features
        self.attn_scale = nn.Parameter(torch.tensor(attn_residual_scale, dtype=torch.float))

        n_nonprice = n_features - 1
        # Base utility on non-price features → scalar
        self.base_net = nn.Sequential(
            nn.Linear(n_nonprice, base_hidden), nn.ReLU(),
            nn.Linear(base_hidden, base_hidden), nn.ReLU(),
            nn.Linear(base_hidden, 1)
        )
        # Monotone price utility
        self.price_net = MonotonePriceNet(n_terms=price_terms)

        # Representation for attention uses non-price features
        self.feature_proj = nn.Linear(n_nonprice, d_model)
        self.attn = MultiHeadSimilarityAttention(d_in=d_model, d_model=d_model, n_heads=n_heads, topk=topk)

        # Map attention features to scalar interaction utility
        self.interaction_net = nn.Sequential(
            nn.Linear(d_model, interaction_hidden), nn.ReLU(),
            nn.Linear(interaction_hidden, 1)
        )

        # Outside option utility network (context: mean of non-price features)
        self.outside_net = nn.Sequential(
            nn.Linear(n_nonprice, outside_hidden), nn.ReLU(),
            nn.Linear(outside_hidden, 1)
        )
        # Bias for outside option
        self.outside_bias = nn.Parameter(torch.tensor(0.0))

    def split_features(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split full features into (nonprice_X, price). X: (B, N, F)."""
        price = X[..., self.price_index]
        if self.price_index == 0:
            nonprice = X[..., 1:]
        elif self.price_index == self.n_features - 1:
            nonprice = X[..., :self.price_index]
        else:
            nonprice = torch.cat([X[..., :self.price_index], X[..., self.price_index + 1:]], dim=-1)
        return nonprice, price

    def forward(
        self,
        X: torch.Tensor,               # (B, N, F) features including price feature (scaled)
        sim: torch.Tensor,             # (N, N) or (B, N, N) similarity (non-negative)
        return_utilities: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, F = X.shape
        assert F == self.n_features
        nonprice, price = self.split_features(X)

        base_u = self.base_net(nonprice).squeeze(-1)           # (B, N)
        price_u = self.price_net(price)                         # (B, N)

        # Attention interaction residual
        z = self.feature_proj(nonprice)                         # (B, N, d_model)
        attn_out = self.attn(z, sim)                           # (B, N, d_model)
        inter_u = self.interaction_net(attn_out).squeeze(-1)   # (B, N)

        utilities = base_u + price_u + self.attn_scale * inter_u  # (B, N)

        # Outside option utility from context (mean of non-price features)
        context = nonprice.mean(dim=1)                          # (B, n_nonprice)
        u0 = self.outside_net(context).squeeze(-1) + self.outside_bias  # (B,)

        # Concatenate outside option as last item index N
        U_full = torch.cat([utilities, u0.unsqueeze(-1)], dim=-1)  # (B, N+1)
        probs = F.softmax(U_full, dim=-1)                           # (B, N+1)

        if return_utilities:
            return probs, U_full
        return probs, None

    # -----------------------------
    # Training loss
    # -----------------------------
    def nll(self, X: torch.Tensor, sim: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood over choices.
        y: (B,) int indices in [0, N] where N denotes the outside option.
        """
        probs, _ = self.forward(X, sim, return_utilities=False)
        logp = torch.log(torch.gather(probs, dim=-1, index=y.view(-1, 1)).squeeze(1).clamp_min(1e-12))
        return -logp.mean()

    # -----------------------------
    # Elasticities and Jacobians (batch=1 for clarity)
    # -----------------------------
    def choice_probs(self, X: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        p, _ = self.forward(X, sim, return_utilities=False)
        return p

    @torch.no_grad()
    def sanity_checks(self, X: torch.Tensor, sim: torch.Tensor, price_grid: Optional[torch.Tensor] = None) -> dict:
        """Basic diagnostics for monotonicity (batch=1)."""
        self.eval()
        B, N, _ = X.shape
        assert B == 1, "Run sanity checks with batch size 1 for simplicity."
        nonprice, price = self.split_features(X)

        # own-price monotonicity via finite differences
        if price_grid is None:
            price_grid = torch.linspace(0.0, 1.0, 21, device=X.device)
        own_monotone = []
        for i in range(N):
            Xtmp = X.repeat(price_grid.numel(), 1, 1)
            Xtmp[:, i, self.price_index] = price_grid
            probs, _ = self.forward(Xtmp, sim, return_utilities=False)
            s_i = probs[:, i]  # share of item i
            diffs = (s_i[1:] - s_i[:-1]).cpu().numpy()
            own_monotone.append(bool((diffs <= 1e-6).all()))
        return {
            "own_price_monotone_all": all(own_monotone),
            "own_price_monotone_per_sku": own_monotone,
        }

    def elasticity_matrix(self, X: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """Compute E in R^{N x N} for batch=1: E_ij = d s_i / d p_j * (p_j / s_i).
        Uses autograd Jacobian. Assumes prices are in X[..., price_index].
        """
        assert X.shape[0] == 1, "Compute elasticities with batch size 1 for clarity."
        X = X.clone().requires_grad_(True)
        probs, _ = self.forward(X, sim, return_utilities=False)  # (1, N+1)
        probs_items = probs[..., :-1].squeeze(0)  # (N,)
        s = probs_items
        N_items = s.shape[0]

        # Build gradients ds/dp via vector-Jacobian products over each SKU i
        grads = []
        for i in range(N_items):
            self.zero_grad(set_to_none=True)
            grad_outputs = torch.zeros_like(probs)
            grad_outputs[0, i] = 1.0
            g = torch.autograd.grad(
                outputs=probs, inputs=X, grad_outputs=grad_outputs,
                retain_graph=True, create_graph=False, allow_unused=True
            )[0]  # (1, N, F)
            dpi = g[0, :, self.price_index]  # (N,)
            grads.append(dpi)
        # grads: list of (N,) where element i is d s_i / d p_j over j
        J = torch.stack(grads, dim=0)  # (N, N)

        price = X[0, :, self.price_index].detach()
        s_safe = s.clamp_min(1e-8)
        E = J * (price.unsqueeze(0) / s_safe.unsqueeze(1))
        return E  # (N, N)

    # -----------------------------
    # Joint price optimization (projected gradient ascent with trust region)
    # -----------------------------
    def optimize_prices(
        self,
        X: torch.Tensor,
        sim: torch.Tensor,
        price_bounds: Tuple[float, float] = (0.0, 1.0),
        steps: int = 50,
        step_size: float = 0.1,
        trust_region: float = 0.05,
        smooth_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize prices (scaled) to maximize revenue R = sum_i p_i * s_i(p).
        X: (1, N, F), prices in column price_index will be updated.
        sim: (N, N) or (1, N, N)
        price_bounds: (lo, hi) in scaled units
        trust_region: max per-step L_inf change of prices
        smooth_penalty: lambda * ||L p||^2 where L is graph Laplacian from sim
        returns: (prices_opt, revenue_series)
        """
        assert X.shape[0] == 1
        lo, hi = price_bounds
        Xopt = X.clone()
        p = Xopt[0, :, self.price_index]
        p = p.clamp(lo, hi).detach().clone().requires_grad_(True)

        # Precompute Laplacian for smoothing if requested
        if smooth_penalty > 0.0:
            if sim.dim() == 3:
                S = sim[0].detach()
            else:
                S = sim.detach()
            S = torch.clamp(S, min=0.0)
            S = row_normalize(S)
            D = torch.diag(S.sum(dim=-1))
            L = D - S  # random-walk Laplacian-like
        else:
            L = None

        rev_series = []
        for t in range(steps):
            Xopt = Xopt.detach().clone()
            Xopt[0, :, self.price_index] = p

            probs, _ = self.forward(Xopt, sim, return_utilities=False)
            s_items = probs[0, :-1]  # (N,)
            revenue = (p * s_items).sum()
            if smooth_penalty > 0.0:
                smooth = (L @ p).pow(2).sum()
                revenue = revenue - smooth_penalty * smooth

            # gradient ascent
            (revenue).backward()
            with torch.no_grad():
                g = p.grad
                if g is None:
                    break
                step = step_size * g
                # trust region (L_inf clip)
                step = torch.clamp(step, min=-trust_region, max=trust_region)
                p.add_(step)
                p.clamp_(lo, hi)
                p.grad = None
                rev_series.append(revenue.detach().cpu())

        Xopt = Xopt.detach().clone()
        Xopt[0, :, self.price_index] = p.detach()
        return Xopt[0, :, self.price_index].detach(), torch.stack(rev_series) if rev_series else torch.tensor([])


# -----------------------------
# Example usage (synthetic)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, F = 1, 8, 5
    price_index = 0  # assume first feature is price (scaled 0..1)

    # Synthetic features: price + 4 attributes
    X = torch.rand(B, N, F)
    X[..., price_index] = torch.linspace(0.2, 0.8, N).view(1, N)

    # Similarity: clustered items
    coords = torch.randn(N, 2)
    dist = torch.cdist(coords, coords)
    S = torch.exp(-dist)

    model = SimilarityAwareChoiceModelImproved(
        n_features=F, price_index=price_index, d_model=64, n_heads=4, topk=4,
        base_hidden=64, interaction_hidden=64, price_terms=8, outside_hidden=32
    )

    # Forward
    probs, U = model(X, S, return_utilities=True)
    print("Choice probs (first 3 items, outside):", probs[0, :3].tolist(), probs[0, -1].item())

    # Sanity check
    checks = model.sanity_checks(X, S)
    print("Own-price monotone all?", checks["own_price_monotone_all"]) 

    # Elasticity matrix
    E = model.elasticity_matrix(X, S)
    print("Elasticity diag (own-price) should be negative:", E.diag())

    # Joint price optimization
    p_opt, rev = model.optimize_prices(X, S, price_bounds=(0.1, 0.9), steps=20, step_size=0.5, trust_region=0.05, smooth_penalty=0.01)
    print("Optimized prices:", p_opt)
    print("Revenue series length:", len(rev))
