"""PSIAL with Pattern-Guided Agent Attention (PGAA).

Requirements: Python >= 3.10, NumPy, PyTorch.
Input:  [B, F, N, T], where F=0 is traffic flow, F=1 is time of day
        in [0, 1), and F=2 is day of week in {0, ..., 6}.
Output: [B, H, N, 1].
Patterns: a [K, L] .npy array constructed using the training set only.

The shared branch uses a Top-1 NCC-matched affine reconstruction;
the private branch uses the residual. Both directions of PGAA use the same
pattern-based node-to-agent collection and query-based agent-to-node delivery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

PathLike = Union[str, Path]


class SpatialFeatureEncoder(nn.Module):
    """Feature enhancement for the shared branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))


class SpatialFeatureEncoder2(nn.Module):
    """Node-channel-aware feature enhancement for the private branch."""

    def __init__(self, num_nodes: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.node_channel_scale = nn.Parameter(torch.ones(1, channels, num_nodes, 1))

    def forward(self, x: Tensor) -> Tensor:
        values = self.dropout(torch.sigmoid(self.value(x)))
        gates = torch.tanh(self.dropout(self.gate(x)) * self.node_channel_scale)
        return values * gates


class TemporalContextEmbedding(nn.Module):
    """Calendar context at the prediction origin."""

    def __init__(self, slots_per_day: int, channels: int) -> None:
        super().__init__()
        self.slots_per_day = slots_per_day
        self.time_of_day_embedding = nn.Parameter(torch.empty(slots_per_day, channels))
        self.weekday_embedding = nn.Parameter(torch.empty(7, channels))
        nn.init.xavier_uniform_(self.time_of_day_embedding)
        nn.init.xavier_uniform_(self.weekday_embedding)

    def forward(self, history_data: Tensor) -> Tensor:
        # history_data: [B,F,N,T]; returns [B,C,N,1].
        tod = history_data[:, 1, :, -1]
        weekday = history_data[:, 2, :, -1]
        tod_idx = torch.floor(tod * self.slots_per_day).long().clamp(0, self.slots_per_day - 1)
        weekday_idx = weekday.long().clamp(0, 6)
        context = self.time_of_day_embedding[tod_idx] + self.weekday_embedding[weekday_idx]
        return context.permute(0, 2, 1).unsqueeze(-1)


class PatternSharedPrivateDecomposition(nn.Module):
    """Zero-lag NCC Top-1 matching, affine shared reconstruction, private residual."""

    def __init__(self, pattern_file: PathLike, input_length: int, eps: float = 1e-5) -> None:
        super().__init__()
        patterns = np.load(Path(pattern_file), allow_pickle=False)
        if not isinstance(patterns, np.ndarray):
            raise ValueError('pattern_file must point to a .npy array.')
        patterns = np.squeeze(patterns)
        if patterns.ndim == 1:
            patterns = patterns[None, :]
        if patterns.ndim != 2 or min(patterns.shape) < 1 or patterns.shape[1] < 2:
            raise ValueError('Expected pattern array [K,L] with K>=1 and L>=2.')
        if not np.issubdtype(patterns.dtype, np.number) or np.iscomplexobj(patterns):
            raise ValueError('Pattern values must be real numbers.')
        patterns = np.asarray(patterns, dtype=np.float32)
        if not np.isfinite(patterns).all():
            raise ValueError('Pattern values must be finite.')

        self.input_length = input_length
        self.pattern_length = patterns.shape[1]
        self.num_patterns = patterns.shape[0]
        self.eps = eps
        full = torch.from_numpy(patterns.copy())
        self.register_buffer('pattern_features', full)
        if self.pattern_length == input_length:
            resized = full.clone()
        else:
            resized = F.interpolate(
                full.unsqueeze(1), size=input_length, mode='linear', align_corners=False
            ).squeeze(1)
        self.register_buffer('input_patterns', resized)

    def _normalize(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / std.clamp_min(self.eps)

    def _routing_history(self, history: Tensor, num_nodes: int) -> Tensor:
        if history.ndim == 4:
            history = history[:, 0]
        if history.ndim != 3:
            raise ValueError('routing_history must have shape [B,N,L], [B,L,N] or [B,F,N,L].')
        if history.shape[1] == num_nodes:
            result = history
        elif history.shape[2] == num_nodes:
            result = history.transpose(1, 2)
        else:
            raise ValueError('Cannot identify node dimension in routing_history.')
        if result.shape[-1] != self.pattern_length:
            raise ValueError(f'routing_history must have length {self.pattern_length}.')
        return result

    def forward(
        self, flow_history: Tensor, routing_history: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Input [B,T,N]; output shared [B,T,N], private [B,T,N], NCC [B,N,K]."""
        with torch.autocast(device_type=flow_history.device.type, enabled=False):
            x = flow_history.float().transpose(1, 2)  # [B,N,T]
            if x.shape[-1] != self.input_length:
                raise ValueError(f'Expected flow history of length {self.input_length}.')
            if routing_history is None:
                routing = x
                patterns = self.input_patterns
            else:
                if self.pattern_length < self.input_length:
                    raise ValueError('Full routing history must cover at least input_length.')
                routing = self._routing_history(routing_history.float(), x.shape[1])
                patterns = self.pattern_features
                if routing.shape[0] != x.shape[0]:
                    raise ValueError('routing_history batch size mismatch.')

            normalized_nodes = self._normalize(routing)
            normalized_patterns = self._normalize(patterns)
            correlation = torch.einsum('bnl,kl->bnk', normalized_nodes, normalized_patterns)
            node_norm = torch.linalg.vector_norm(normalized_nodes, dim=-1)
            pattern_norm = torch.linalg.vector_norm(normalized_patterns, dim=-1)
            norm_product = (node_norm.unsqueeze(-1) * pattern_norm[None, None]).clamp_min(self.eps)
            scores = (correlation / norm_product).clamp(-1.0, 1.0)  # [B,N,K]

            # Hard Top-1 matching ONLY for shared-component reconstruction.
            top_index = scores.argmax(dim=-1)                 # [B,N]
            matched = normalized_patterns[top_index]         # [B,N,L]
            node_mean = routing.mean(dim=-1, keepdim=True)
            centered_node = routing - node_mean
            pattern_mean = matched.mean(dim=-1, keepdim=True)
            centered_pattern = matched - pattern_mean
            covariance = (centered_node * centered_pattern).sum(dim=-1, keepdim=True)
            energy = centered_pattern.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
            scale = (covariance / energy).clamp_min(0.0)
            shift = node_mean - scale * pattern_mean
            shared = scale * matched[..., -self.input_length:] + shift
            private = x - shared
            return shared.transpose(1, 2), private.transpose(1, 2), scores


class TemporalPatchMerge(nn.Module):
    """Learned temporal patch aggregation."""

    def __init__(self, channels: int, patch_size: int, dropout: float) -> None:
        super().__init__()
        self.merge = nn.Conv2d(
            channels, channels, kernel_size=(1, patch_size), stride=(1, patch_size)
        )
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.norm(self.merge(x))))


class PatternGuidedAgentAttention(nn.Module):
    """Pattern-Guided Agent Attention (PGAA): bidirectional shared/private exchange.

    Pattern-based collection: C = softmax_N(log R^T),   [B,M,N]
    Agent-to-node delivery: D = softmax_M(Q A^T / sqrt(d)).
    Message: D @ (C @ V), with an independent feature gate in each direction.

    Both directions use the same pattern collection C. No Key projection and
    no dense N-by-N attention matrix are required.
    """

    def __init__(
        self, channels: int, num_agents: int, num_heads: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1 or num_agents < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError('channels must be divisible by num_heads.')
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.shared_norm = nn.LayerNorm(channels, eps=1e-6)
        self.private_norm = nn.LayerNorm(channels, eps=1e-6)
        # Only Query and Value are used; Key projection is intentionally removed.
        self.shared_qv = nn.Linear(channels, 2 * channels, bias=False)
        self.private_qv = nn.Linear(channels, 2 * channels, bias=False)

        self.agents_to_shared = nn.Parameter(torch.empty(num_heads, num_agents, self.head_dim))
        self.agents_to_private = nn.Parameter(torch.empty(num_heads, num_agents, self.head_dim))
        nn.init.normal_(self.agents_to_shared, std=0.02)
        nn.init.normal_(self.agents_to_private, std=0.02)

        self.output_to_shared = nn.Linear(channels, channels, bias=False)
        self.output_to_private = nn.Linear(channels, channels, bias=False)
        nn.init.xavier_uniform_(self.output_to_shared.weight, gain=0.1)
        nn.init.xavier_uniform_(self.output_to_private.weight, gain=0.1)

        self.gate_to_shared = nn.Linear(3 * channels, channels)
        self.gate_to_private = nn.Linear(3 * channels, channels)
        nn.init.constant_(self.gate_to_shared.bias, -2.0)
        nn.init.constant_(self.gate_to_private.bias, -2.0)
        self.message_dropout = nn.Dropout(dropout)

    def _qv(self, x: Tensor, projection: nn.Linear) -> tuple[Tensor, Tensor]:
        bt, nodes, _ = x.shape
        qv = projection(x).reshape(bt, nodes, 2, self.num_heads, self.head_dim)
        q, v = qv.permute(2, 0, 3, 1, 4).unbind(0)
        return q, v

    def _message(self, q: Tensor, v: Tensor, agents: Tensor, collect: Tensor) -> Tensor:
        # q/v: [B*T,H,N,d]; collect: [B*T,1,M,N].
        with torch.autocast(device_type=q.device.type, enabled=False):
            a = agents.float().unsqueeze(0)  # [1,H,M,d]
            agent_values = collect.float() @ v.float()
            distribute = torch.softmax((q.float() @ a.transpose(-2, -1)) * self.scale, dim=-1)
            message = distribute @ agent_values
        return message.to(v.dtype).transpose(1, 2).reshape(q.shape[0], q.shape[2], self.channels)

    def forward(
        self, shared: Tensor, private: Tensor, collect_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # shared/private: [B,C,N,T]; collect_weights: [B,M,N].
        batch, channels, nodes, times = shared.shape

        def tokens(x: Tensor) -> Tensor:
            return x.permute(0, 3, 2, 1).reshape(batch * times, nodes, channels)

        def spatial(x: Tensor) -> Tensor:
            return x.reshape(batch, times, nodes, channels).permute(0, 3, 2, 1).contiguous()

        collect = collect_weights[:, None].expand(batch, times, -1, -1)
        collect = collect.reshape(batch * times, 1, collect.shape[2], nodes)

        s, p = tokens(shared), tokens(private)
        sn, pn = self.shared_norm(s), self.private_norm(p)
        qs, vs = self._qv(sn, self.shared_qv)
        qp, vp = self._qv(pn, self.private_qv)

        # Both directions read the SAME states before residual updates.
        ms = self.output_to_shared(self._message(qs, vp, self.agents_to_shared, collect))
        mp = self.output_to_private(self._message(qp, vs, self.agents_to_private, collect))
        gs = torch.sigmoid(self.gate_to_shared(torch.cat((sn, pn, ms), dim=-1)))
        gp = torch.sigmoid(self.gate_to_private(torch.cat((pn, sn, mp), dim=-1)))
        updated_shared = spatial(s + (gs * self.message_dropout(ms)).to(s.dtype))
        updated_private = spatial(p + (gp * self.message_dropout(mp)).to(p.dtype))
        return updated_shared, updated_private


class HierarchicalSpatialTemporalStage(nn.Module):
    """Compress time in both branches, then apply PGAA."""

    def __init__(
        self, channels: int, patch_size: int, num_agents: int,
        num_heads: int, dropout: float,
    ) -> None:
        super().__init__()
        self.shared_patch_merge = TemporalPatchMerge(channels, patch_size, dropout)
        self.private_patch_merge = TemporalPatchMerge(channels, patch_size, dropout)
        self.attention = PatternGuidedAgentAttention(channels, num_agents, num_heads, dropout)

    def forward(
        self, shared: Tensor, private: Tensor, collect_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        shared = self.shared_patch_merge(shared)
        private = self.private_patch_merge(private)
        return self.attention(shared, private, collect_weights)


class HierarchicalSpatialTemporalEncoder(nn.Module):
    """Multi-scale temporal aggregation interleaved with PGAA."""

    def __init__(
        self, channels: int, input_length: int, num_agents: int,
        patch_sizes: Optional[Sequence[int]], num_heads: int, dropout: float,
    ) -> None:
        super().__init__()
        if patch_sizes is None:
            factors = (5, 4, 3, 2)
            remaining = input_length
            sizes = []
            while remaining > 1:
                size = next((p for p in factors if remaining % p == 0), remaining)
                sizes.append(size)
                remaining //= size
            patch_sizes = sizes

        self.patch_sizes = tuple(int(p) for p in patch_sizes)
        if not self.patch_sizes:
            raise ValueError('patch_sizes cannot be empty.')
        remaining = input_length
        for size in self.patch_sizes:
            if size < 2 or remaining % size:
                raise ValueError(f'Invalid patch_size={size} for temporal length {remaining}.')
            remaining //= size
        if remaining != 1:
            raise ValueError('patch_sizes must compress the input length to 1.')

        self.stages = nn.ModuleList([
            HierarchicalSpatialTemporalStage(channels, size, num_agents, num_heads, dropout)
            for size in self.patch_sizes
        ])
        groups = 8 if channels % 8 == 0 else 1
        self.shared_output = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.Conv2d(channels, channels, 1),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.private_output = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.Conv2d(channels, channels, 1),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(
        self, shared: Tensor, private: Tensor, collect_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for stage in self.stages:
            shared, private = stage(shared, private, collect_weights)
        return self.shared_output(shared), self.private_output(private)


def infer_slots_per_day(num_nodes: int) -> int:
    if num_nodes in {170, 307, 358, 883}:
        return 288
    if num_nodes in {250, 266}:
        return 48
    if num_nodes > 200:
        return 96
    raise ValueError('Unknown data sampling rate; pass slots_per_day explicitly.')


class PSIAL(nn.Module):
    """Full PSIAL model, using PGAA in both directions at each temporal scale."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        input_dim: int = 3,
        channels: int = 64,
        num_nodes: int = 170,
        input_len: int = 12,
        output_len: int = 12,
        dropout: float = 0.1,
        pattern_file: PathLike = 'data/PEMS08/K32_L12_centers.npy',
        slots_per_day: Optional[int] = None,
        patch_sizes: Optional[Sequence[int]] = None,
        pattern_temperature: float = 0.1,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if input_dim < 3 or input_len < 2 or num_nodes < 1 or output_len < 1:
            raise ValueError('Require input_dim>=3, input_len>=2, and positive nodes/output_len.')
        if num_heads < 1 or channels < 2 or channels % 2 or (channels // 2) % num_heads:
            raise ValueError('channels must be even and channels//2 must divide by num_heads.')
        if not np.isfinite(pattern_temperature) or pattern_temperature <= 0:
            raise ValueError('pattern_temperature must be positive.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_length = input_len
        self.output_length = output_len
        self.pattern_temperature = pattern_temperature

        self.pattern_decomposition = PatternSharedPrivateDecomposition(pattern_file, input_len)
        num_agents = self.pattern_decomposition.num_patterns  # One Agent per pattern.
        half = channels // 2
        self.shared_input_projection = nn.Conv2d(1, half, 1)
        self.private_input_projection = nn.Conv2d(1, half, 1)
        slots = infer_slots_per_day(num_nodes) if slots_per_day is None else int(slots_per_day)
        if slots < 1:
            raise ValueError('slots_per_day must be positive.')
        self.temporal_context_embedding = TemporalContextEmbedding(slots, channels)
        self.temporal_compressor = HierarchicalSpatialTemporalEncoder(
            half, input_len, num_agents, patch_sizes, num_heads, dropout
        )
        self.spatial_encoder = SpatialFeatureEncoder(2 * channels)
        self.spatial_encoder2 = SpatialFeatureEncoder2(num_nodes, 2 * channels, dropout)
        self.output_projection = nn.Conv2d(2 * channels, output_len, 1)
        if device is not None:
            self.to(device)

    def forward(self, history_data: Tensor, routing_history: Optional[Tensor] = None) -> Tensor:
        """Predict [B,H,N,1] from [B,F,N,T].

        Optional routing_history must end at the same prediction origin and
        have temporal length L equal to the pattern file's temporal length.
        """
        expected = (self.input_dim, self.num_nodes, self.input_length)
        if history_data.ndim != 4 or tuple(history_data.shape[1:]) != expected:
            raise ValueError(f'Expected [B,{expected[0]},{expected[1]},{expected[2]}], '
                             f'got {tuple(history_data.shape)}.')
        if history_data.shape[0] < 1 or not history_data.is_floating_point():
            raise ValueError('history_data must be a nonempty floating-point batch.')

        shared_history, private_history, scores = self.pattern_decomposition(
            history_data[:, 0].transpose(1, 2), routing_history
        )
        shared = self.shared_input_projection(shared_history.transpose(1, 2).unsqueeze(1))
        private = self.private_input_projection(private_history.transpose(1, 2).unsqueeze(1))

        # Each row softmax_K produces node-to-pattern soft memberships.
        # Each column softmax_N produces pattern-wise node collection weights.
        with torch.autocast(device_type=scores.device.type, enabled=False):
            log_prior = F.log_softmax(scores.float() / self.pattern_temperature, dim=-1)
            collect_weights = torch.softmax(log_prior.transpose(-2, -1), dim=-1)  # [B,K,N]

        shared_state, private_state = self.temporal_compressor(
            shared, private, collect_weights
        )
        context = self.temporal_context_embedding(history_data)
        shared_features = self.spatial_encoder(
            torch.cat((shared_state, shared[..., -1:], context), dim=1)
        )
        private_features = self.spatial_encoder2(
            torch.cat((private_state, private[..., -1:], context), dim=1)
        )
        return self.output_projection(shared_features + private_features)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters())





def _smoke_test() -> None:
    import tempfile

    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as folder:
        pattern_file = Path(folder) / 'patterns.npy'
        np.save(pattern_file, np.random.default_rng(7).normal(size=(8, 12)).astype(np.float32))
        model = PSIAL(num_nodes=10, channels=32, pattern_file=pattern_file,
                      slots_per_day=288, num_heads=2)
        x = torch.randn(2, 3, 10, 12)
        x[:, 1] = torch.rand(2, 10, 12)
        x[:, 2] = torch.randint(0, 7, (2, 10, 12)).float()
        y = model(x)
        assert y.shape == (2, 12, 10, 1) and torch.isfinite(y).all()
        y.square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        print('PASS: forward/backward; output:', tuple(y.shape))
        print('Parameters:', model.param_num())


if __name__ == '__main__':
    _smoke_test()
