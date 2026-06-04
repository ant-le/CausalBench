from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model


def _set_diagonal(arr: torch.Tensor, value: float) -> torch.Tensor:
    n_vars = arr.shape[-1]
    diag_idx = torch.arange(n_vars, device=arr.device)
    out = arr.clone()
    out[..., diag_idx, diag_idx] = value
    return out


class _AviciAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        key_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * key_size
        self.num_heads = num_heads
        self.key_size = key_size
        self.q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.v_proj = nn.Linear(dim, inner_dim, bias=True)
        self.out_proj = nn.Linear(inner_dim, dim, bias=True)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q_shape = q.shape
        seq_len = q_shape[-2]
        flat_batch = int(math.prod(q_shape[:-2]))
        q_flat = q.reshape(flat_batch, seq_len, q_shape[-1])
        k_flat = k.reshape(flat_batch, seq_len, k.shape[-1])
        v_flat = v.reshape(flat_batch, seq_len, v.shape[-1])

        q_proj = self.q_proj(q_flat).view(
            flat_batch, seq_len, self.num_heads, self.key_size
        )
        k_proj = self.k_proj(k_flat).view(
            flat_batch, seq_len, self.num_heads, self.key_size
        )
        v_proj = self.v_proj(v_flat).view(
            flat_batch, seq_len, self.num_heads, self.key_size
        )

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.key_size)
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-1, -2)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v_proj)

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(flat_batch, seq_len, self.num_heads * self.key_size)
        )
        out = self.out_proj(attn_out)
        return out.reshape(*q_shape[:-2], seq_len, out.shape[-1])


class _AviciBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        key_size: int,
        num_heads: int,
        widening_factor: int,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_k = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)
        self.attn = _AviciAttention(dim=dim, key_size=key_size, num_heads=num_heads)
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, widening_factor * dim),
            nn.ReLU(),
            nn.Linear(widening_factor * dim, dim),
        )

    def forward(self, z: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        q_in = self.ln_q(z)
        k_in = self.ln_k(z)
        v_in = self.ln_v(z)
        z_attn = self.attn(q_in, k_in, v_in)
        z = z + F.dropout(z_attn, p=dropout_rate, training=self.training)

        z_in = self.ffn_ln(z)
        z_ffn = self.ffn(z_in)
        z = z + F.dropout(z_ffn, p=dropout_rate, training=self.training)
        return z


@register_model("avici")
class AviciModel(BaseModel):
    def __init__(
        self,
        num_nodes: int,
        dim: int = 128,
        layers: int = 8,
        key_size: int = 32,
        num_heads: int = 8,
        widening_factor: int = 4,
        dropout: float = 0.1,
        out_dim: int | None = None,
        logit_bias_init: float = -3.0,
        cosine_temp_init: float = 0.0,
        mask_diag: bool = True,
        acyclicity_weight: float | None = 0.0,
        acyclicity_pow_iters: int = 10,
        regulariser_lr: float = 1e-4,
        regulariser_ema_alpha: float = 0.05,
        regulariser_warmup_updates: int = 100,
        regulariser_max_weight: float | None = None,
        experimental_chunk_size: int | None = None,
        d_model: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
        dim_feedforward: int | None = None,
        emb_depth: int | None = None,
        use_positional_encoding: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__()

        if d_model is not None:
            dim = d_model
        if nhead is not None:
            num_heads = nhead
        if num_layers is not None:
            layers = num_layers
        if dim_feedforward is not None and dim > 0:
            widening_factor = max(1, int(dim_feedforward // dim))

        del emb_depth
        del use_positional_encoding

        self.num_nodes = num_nodes
        self.dim = dim
        self.d_model = dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.key_size = key_size
        self.num_heads = num_heads
        self.widening_factor = widening_factor
        self.logit_bias_init = logit_bias_init
        self.cosine_temp_init = cosine_temp_init
        self.mask_diag = mask_diag
        self.acyclicity_weight = acyclicity_weight
        self.acyclicity_pow_iters = acyclicity_pow_iters
        self.regulariser_lr = regulariser_lr
        self.regulariser_ema_alpha = regulariser_ema_alpha
        self.regulariser_warmup_updates = regulariser_warmup_updates
        self.regulariser_max_weight = regulariser_max_weight
        self.experimental_chunk_size = experimental_chunk_size

        self.input_proj = nn.Linear(2, dim)
        self.blocks = nn.ModuleList(
            [
                _AviciBlock(
                    dim=dim,
                    key_size=key_size,
                    num_heads=num_heads,
                    widening_factor=widening_factor,
                )
                for _ in range(self.layers)
            ]
        )
        self.final_ln = nn.LayerNorm(dim)
        self.u_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.out_dim))
        self.v_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.out_dim))
        self.learned_temp = nn.Parameter(
            torch.tensor(float(cosine_temp_init), dtype=torch.float32)
        )
        self.final_matrix_bias = nn.Parameter(
            torch.tensor(float(logit_bias_init), dtype=torch.float32)
        )
        self.register_buffer(
            "regulariser_weight",
            torch.tensor(float(acyclicity_weight or 0.0), dtype=torch.float32),
        )
        self.register_buffer("regulariser_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer(
            "regulariser_update_count",
            torch.tensor(0, dtype=torch.long),
        )
        self.regulariser_weight: torch.Tensor
        self.regulariser_ema: torch.Tensor
        self.regulariser_update_count: torch.Tensor

    @property
    def needs_pretraining(self) -> bool:
        return True

    def _prepare_input(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            zeros = torch.zeros_like(input_data)
            return torch.stack([input_data, zeros], dim=-1)
        if input_data.ndim == 4:
            if input_data.size(-1) == 1:
                zeros = torch.zeros_like(input_data)
                return torch.cat([input_data, zeros], dim=-1)
            if input_data.size(-1) >= 2:
                return input_data[..., :2]
        raise ValueError("AVICI input must have shape (B, N, d) or (B, N, d, c).")

    def _all_blocks_and_max(
        self, z: torch.Tensor, *, dropout_rate: float
    ) -> torch.Tensor:
        if self.layers % 2 != 0:
            raise RuntimeError("Number of AVICI layers must be even.")
        for block in self.blocks:
            z = block(z, dropout_rate)
            z = z.swapaxes(-3, -2)
        z = self.final_ln(z)
        return z.max(dim=-3).values

    def _forward_embedded(
        self, z: torch.Tensor, *, dropout_rate: float
    ) -> torch.Tensor:
        n_obs = z.shape[-3]
        chunk_size = self.experimental_chunk_size
        if chunk_size is not None and 0 < chunk_size < n_obs:
            if n_obs % chunk_size != 0:
                raise ValueError(
                    "observations axis must be divisible by experimental_chunk_size"
                )
            chunks = n_obs // chunk_size
            z = z.reshape(*z.shape[:-3], chunk_size, chunks, *z.shape[-2:])
            z = z.swapaxes(-3, 0)
            z = torch.stack(
                [
                    self._all_blocks_and_max(chunk, dropout_rate=dropout_rate)
                    for chunk in z
                ],
                dim=0,
            )
            return z.max(dim=0).values
        return self._all_blocks_and_max(z, dropout_rate=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_mask = (
            mask.to(device=x.device, dtype=torch.bool) if mask is not None else None
        )
        x_prepared = self._prepare_input(x)
        if node_mask is not None:
            x_prepared = x_prepared.masked_fill(
                node_mask.unsqueeze(1).unsqueeze(-1),
                0.0,
            )

        dropout_rate = self.dropout if self.training else 0.0
        z = self.input_proj(x_prepared)
        z = self._forward_embedded(z, dropout_rate=dropout_rate)

        u = self.u_proj(z)
        v = self.v_proj(z)
        u = F.normalize(u, p=2, dim=-1, eps=1e-12)
        v = F.normalize(v, p=2, dim=-1, eps=1e-12)

        logits = torch.einsum("...id,...jd->...ij", u, v)
        logits = logits * self.learned_temp.exp()
        logits = logits + self.final_matrix_bias
        if node_mask is not None:
            pad_edges = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
            logits = logits.masked_fill(pad_edges, -20.0)
        return logits

    def predict_probabilities(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, mask=mask)
        probs = torch.sigmoid(logits)
        if mask is not None:
            node_mask = mask.to(device=probs.device, dtype=torch.bool)
            pad_edges = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
            probs = probs.masked_fill(pad_edges, 0.0)
        if self.mask_diag:
            probs = _set_diagonal(probs, 0.0)
        return probs

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = self.predict_probabilities(x, mask=mask)
        samples = torch.bernoulli(
            probs.unsqueeze(1).expand(-1, num_samples, -1, -1)
        ).to(dtype=torch.int32)
        return samples.to(dtype=probs.dtype)

    def _exp_matmul(
        self, logmat: torch.Tensor, vec: torch.Tensor, axis: int
    ) -> torch.Tensor:
        mat = torch.exp(logmat)
        if axis == -1:
            return torch.einsum("...ij,...j->...i", mat, vec)
        if axis == -2:
            return torch.einsum("...i,...ij->...j", vec, mat)
        raise ValueError("axis must be -1 or -2")

    def _acyclicity_spectral_log(self, logmat: torch.Tensor) -> torch.Tensor:
        # Force full-precision arithmetic.  Under AMP autocast the einsum ops
        # in the power iteration are silently downcast to bf16/fp16, which
        # causes catastrophic numerical instability (loss diverging to -1e9).
        with torch.autocast(device_type=logmat.device.type, enabled=False):
            logmat = logmat.float()
            u = torch.rand(logmat.shape[:-1], device=logmat.device, dtype=torch.float32)
            v = torch.rand(logmat.shape[:-1], device=logmat.device, dtype=torch.float32)

            for _ in range(self.acyclicity_pow_iters):
                u_new = self._exp_matmul(logmat, u, -2)
                v_new = self._exp_matmul(logmat, v, -1)
                u = u_new / u_new.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                v = v_new / v_new.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

            u = u.detach()
            v = v.detach()
            numerator = torch.einsum(
                "...j,...j->...", u, self._exp_matmul(logmat, v, -1)
            )
            denominator = torch.einsum("...j,...j->...", u, v).clamp_min(1e-12)
            return numerator / denominator

    def _reduce_regulariser_signal(self, penalty: torch.Tensor) -> torch.Tensor:
        reduced = penalty.detach().to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced = reduced / float(dist.get_world_size())
        return reduced

    def update_regulariser_weight(self, acyclic_penalty: torch.Tensor) -> None:
        alpha = min(max(float(self.regulariser_ema_alpha), 0.0), 1.0)
        warmup_updates = max(0, int(self.regulariser_warmup_updates))
        max_weight = self.regulariser_max_weight

        with torch.no_grad():
            penalty = self._reduce_regulariser_signal(acyclic_penalty)
            if not torch.isfinite(penalty):
                return

            if alpha >= 1.0 or int(self.regulariser_update_count.item()) == 0:
                self.regulariser_ema.copy_(penalty)
            else:
                self.regulariser_ema.mul_(1.0 - alpha).add_(alpha * penalty)

            next_count = int(self.regulariser_update_count.item()) + 1
            if warmup_updates <= 0:
                warmup_scale = 1.0
            else:
                warmup_scale = min(1.0, next_count / float(warmup_updates))

            self.regulariser_weight.add_(
                float(self.regulariser_lr) * warmup_scale * self.regulariser_ema
            )
            if max_weight is None:
                self.regulariser_weight.clamp_(min=0.0)
            else:
                self.regulariser_weight.clamp_(min=0.0, max=float(max_weight))
            self.regulariser_update_count.fill_(next_count)

    def calculate_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        update_regulariser: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        logits = output.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        logp1 = F.logsigmoid(logits)
        logp0 = F.logsigmoid(-logits)
        loss_eltwise = -(target * logp1 + (1.0 - target) * logp0)

        edge_mask = torch.ones_like(loss_eltwise)
        if self.mask_diag:
            edge_mask = _set_diagonal(edge_mask, 0.0)
        if node_mask is not None:
            node_mask = node_mask.to(device=logits.device, dtype=torch.bool)
            pad_edges = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
            edge_mask = edge_mask.masked_fill(pad_edges, 0.0)

        valid_edges = edge_mask.sum(dim=(-1, -2)).clamp_min(1.0)
        batch_loss = (loss_eltwise * edge_mask).sum(dim=(-1, -2)) / valid_edges
        loss_raw = batch_loss.mean()

        if self.acyclicity_weight is None:
            return loss_raw

        if self.mask_diag:
            logp_edges = _set_diagonal(logp1, float("-inf"))
        else:
            logp_edges = logp1
        if node_mask is not None:
            pad_edges = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
            logp_edges = logp_edges.masked_fill(pad_edges, float("-inf"))
        spectral_radii = self._acyclicity_spectral_log(logp_edges)
        ave_acyc_penalty = spectral_radii.mean()
        # Use the current weight as a plain float so that inplace updates in
        # ``update_regulariser_weight`` cannot invalidate the autograd graph.
        wgt_acyc_penalty = float(self.regulariser_weight.item()) * ave_acyc_penalty

        if update_regulariser:
            self.update_regulariser_weight(ave_acyc_penalty)

        return loss_raw + wgt_acyc_penalty
