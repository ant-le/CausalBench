from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model
from causal_meta.models.utils.nn import (
    CausalAdjacencyMatrix,
    CausalTNPEncoder,
    CausalTransformerDecoderLayer,
    build_mlp,
)
from causal_meta.models.utils.permutations import sample_permutation

log = logging.getLogger(__name__)


@register_model("bcnp")
class BCNP(CausalTNPEncoder, BaseModel):
    """
    Bayesian Causal Neural Process (BCNP).
    Replicates the CausalProbabilisticDecoder architecture from the reference.
    Generates DAGs via Sinkhorn Permutations: A = P @ L @ P.T
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,  # encoder layers
        num_layers_decoder: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        emb_depth: int = 1,
        use_positional_encoding: bool = False,
        input_dim: int = 2,
        n_perm_samples: int = 10,
        sinkhorn_iter: int = 20,
        q_before_l: bool = True,
        debug_logging: bool = False,
        debug_log_every_n_steps: int = 1000,
        **kwargs,
    ) -> None:
        CausalTNPEncoder.__init__(
            self,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            input_dim=input_dim,
            device=None,
            dtype=None,
            avici_summary=False,  # BCNP uses Attention Summary (default)
        )
        self.d_model = d_model

        self.num_nodes = num_nodes
        self.input_dim = int(input_dim)
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.q_before_l = q_before_l
        self.debug_logging = bool(debug_logging)
        self.debug_log_every_n_steps = max(1, int(debug_log_every_n_steps))
        self._debug_forward_count = 0

        # Decoders
        # The reference splits num_layers_decoder into two halves
        decoder_layers_half = max(1, num_layers_decoder // 2)

        self.decoder_L = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                bias=False,
            ),
            num_layers=decoder_layers_half,
        )

        self.decoder_Q = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                bias=False,
            ),
            num_layers=decoder_layers_half,
        )

        self.L_param = CausalAdjacencyMatrix(
            nhead=nhead, d_model=d_model, device=None, dtype=None
        )

        self.permutation_logit_network = build_mlp(
            dim_in=d_model, dim_hid=d_model, dim_out=1, depth=emb_depth
        )

    def _should_log_debug(self) -> bool:
        if not self.debug_logging:
            return False
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        return (
            self._debug_forward_count == 1
            or self._debug_forward_count % self.debug_log_every_n_steps == 0
        )

    @staticmethod
    def _tensor_debug_stats(name: str, tensor: torch.Tensor) -> str:
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = int(detached.numel())
        nan_count = int(torch.isnan(detached).sum().item())
        inf_count = int(torch.isinf(detached).sum().item())
        if finite_count == 0:
            return (
                f"{name}[shape={tuple(detached.shape)}, finite=0/{total_count}, "
                f"nan={nan_count}, inf={inf_count}]"
            )

        finite = detached[finite_mask]
        return (
            f"{name}[shape={tuple(detached.shape)}, min={float(finite.min().item()):.4f}, "
            f"max={float(finite.max().item()):.4f}, mean={float(finite.mean().item()):.4f}, "
            f"finite={finite_count}/{total_count}, nan={nan_count}, inf={inf_count}]"
        )

    @property
    def needs_pretraining(self) -> bool:
        return True

    def decode(self, representation: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if not self.q_before_l:
            L_rep = self.decoder_L(
                representation, memory=None, tgt_key_padding_mask=mask
            )
            Q_rep = self.decoder_Q(L_rep, memory=None, tgt_key_padding_mask=mask)
        else:
            Q_rep = self.decoder_Q(
                representation, memory=None, tgt_key_padding_mask=mask
            )
            L_rep = self.decoder_L(Q_rep, memory=None, tgt_key_padding_mask=mask)

        # Get L matrix parameters
        L_param = self.L_param(L_rep, padding_mask=mask)
        # Symmetrize L_param for permutation equivariance (as per reference)
        L_param = (L_param + L_param.transpose(1, 2)) / 2

        return L_param, Q_rep

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return edge probabilities marginalized over permutations.

        Args:
            x: Input tensor of shape ``(Batch, Samples, Variables)``.
            mask: Optional padding mask for variable-size graphs.  When
                provided it should be a boolean-like tensor of shape
                ``(Batch, Variables)`` with ``True`` / ``-inf`` at padded
                positions (matching the transformer ``key_padding_mask``
                convention).  Padded nodes will be masked out of the
                permutation logits so that only real nodes participate.

        Returns:
            all_probs: Tensor of shape
                ``(n_perm_samples, Batch, Variables, Variables)``.
        """
        self._debug_forward_count += 1
        input_data = x
        if input_data.ndim == 3:
            values = input_data
            if self.input_dim > 1:
                intervention_mask = torch.zeros_like(values)
                input_data = torch.stack([values, intervention_mask], dim=-1)
            else:
                input_data = values.unsqueeze(-1)
        elif input_data.ndim == 4:
            if input_data.size(-1) >= self.input_dim:
                input_data = input_data[..., : self.input_dim]
            else:
                pad = self.input_dim - int(input_data.size(-1))
                input_data = torch.cat(
                    [
                        input_data,
                        torch.zeros_like(input_data[..., :1]).repeat_interleave(
                            pad, dim=-1
                        ),
                    ],
                    dim=-1,
                )
        else:
            raise ValueError("BCNP input must have shape (B, S, V) or (B, S, V, C).")

        B, S, V, _ = input_data.shape

        if V > self.num_nodes:
            raise ValueError(
                f"Number of input variables ({V}) exceeds the configured "
                f"num_nodes ({self.num_nodes}).  Use V <= num_nodes and "
                f"supply a padding mask for smaller graphs."
            )

        # Build decoder padding mask from the provided mask.  The
        # encoder already accepts ``mask`` directly; the decoder
        # ``tgt_key_padding_mask`` uses the same convention (``-inf``
        # at padded positions, ``0`` elsewhere).
        decoder_mask: Optional[torch.Tensor] = None
        if mask is not None:
            # Normalise to the ``-inf / 0`` convention expected by
            # ``nn.TransformerDecoder.tgt_key_padding_mask``.
            if mask.dtype == torch.bool:
                decoder_mask = torch.zeros_like(mask, dtype=input_data.dtype)
                decoder_mask.masked_fill_(mask, float("-inf"))
            else:
                decoder_mask = mask.to(dtype=input_data.dtype)

        representation = self.encode(target_data=input_data, mask=mask)
        representation = representation.squeeze(2)  # (B, V, D)

        L_param, Q_rep = self.decode(representation, mask=decoder_mask)

        # Calculate Q parameters (permutation logits)
        permutation_logits = self.permutation_logit_network(Q_rep).squeeze(-1)  # (B, V)
        ovector = torch.arange(
            1, V + 1, device=input_data.device, dtype=input_data.dtype
        )

        # Outer product to get matrix logits
        Q_param = torch.einsum("bn,m->bnm", permutation_logits, ovector)
        Q_param = F.logsigmoid(Q_param)

        # Apply padding mask to Q_param when a mask is provided.
        # Reference logic: Q_mask = decoder_mask[:,None,:] + decoder_mask[:,:,None]
        #                  Q_mask *= (1 - eye);  Q_param += Q_mask
        # This drives padded rows/columns of Q_param to -inf so that
        # Sinkhorn normalisation assigns near-zero mass to padded nodes.
        if decoder_mask is not None:
            Q_mask = decoder_mask.unsqueeze(1) + decoder_mask.unsqueeze(2)
            eye_mask = torch.eye(V, device=input_data.device, dtype=torch.bool)
            Q_mask = Q_mask.masked_fill(eye_mask, 0.0)
            Q_param = Q_param + Q_mask

        # Sample Permutations
        perm, _ = sample_permutation(
            log_alpha=Q_param,
            temp=1.0,
            noise_factor=1.0,
            n_samples=self.n_perm_samples,
            hard=True,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            device=input_data.device,
        )
        perm = perm.transpose(1, 0)  # (K, B, N, N)
        perm_inv = perm.transpose(3, 2)

        # Lower Triangular Mask (strictly below diagonal)
        tril_mask = torch.tril(
            torch.ones((V, V), device=input_data.device), diagonal=-1
        )

        # DAG construction: P @ L_mask @ P.T
        all_masks = torch.einsum("nbij,jk,nbkl->nbil", perm, tril_mask, perm_inv)

        # Probs from L
        probs = torch.sigmoid(L_param)  # (B, N, N)

        # Elementwise mul
        all_probs = probs.unsqueeze(0) * all_masks  # (K, B, N, N)

        if self._should_log_debug():
            log.info(
                "BCNP debug: mode=%s, forward=%d, %s, %s, %s",
                "train" if self.training else "eval",
                self._debug_forward_count,
                self._tensor_debug_stats("Q_param", Q_param),
                self._tensor_debug_stats("L_param", L_param),
                self._tensor_debug_stats("all_probs", all_probs),
            )

        return all_probs

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return sampled graphs.

        Args:
            x: Input data of shape ``(Batch, Samples, Variables)``.
            num_samples: Number of graph samples to return per batch element.
            mask: Optional padding mask (see :meth:`forward`).

        Returns:
            Sampled adjacency matrices of shape
            ``(Batch, num_samples, Variables, Variables)``.
        """
        input_data = x
        # We temporarily set n_perm_samples to requested num_samples
        original_k = self.n_perm_samples
        self.n_perm_samples = num_samples

        all_probs = self.forward(input_data, mask=mask)  # (K, B, N, N)

        self.n_perm_samples = original_k

        # Sample Bernoulli
        samples = torch.bernoulli(all_probs)

        # Transpose to (B, K, N, N) to match BaseModel interface
        samples = samples.permute(1, 0, 2, 3)

        return samples

    def calculate_loss(
        self,
        output: Any,
        target: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
        -----
            output: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
        """
        del kwargs
        probs = output
        target_graph = target
        existence_dist = torch.distributions.Bernoulli(
            probs=probs.clamp(1e-6, 1 - 1e-6)
        )
        log_prob = existence_dist.log_prob(target_graph.unsqueeze(0))
        log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
        loss_per_edge = -log_prob_sum

        batch_size, n_nodes, _ = target_graph.shape
        edge_mask = torch.ones(
            batch_size,
            n_nodes,
            n_nodes,
            device=target_graph.device,
            dtype=target_graph.dtype,
        )
        diag = torch.eye(
            n_nodes, device=target_graph.device, dtype=torch.bool
        ).unsqueeze(0)
        edge_mask = edge_mask.masked_fill(diag, 0.0)

        if node_mask is not None:
            node_mask = node_mask.to(device=target_graph.device, dtype=torch.bool)
            pad_edges = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
            edge_mask = edge_mask.masked_fill(pad_edges, 0.0)

        valid_edges = edge_mask.sum(dim=(-1, -2)).clamp_min(1.0)
        loss = (loss_per_edge * edge_mask).sum(dim=(-1, -2)) / valid_edges
        return loss.mean()
