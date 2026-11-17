"""The codec on Megatron's Flex+DeepEP dispatcher: what each MoE layer sends between
expert-parallel ranks during training.

On dispatch the sender compresses its rows (a reference layer instead all-gathers its input to
every rank and sends an empty row) and the receiver decompresses, or a folded student's folded
input projections W_in,i (`readers`) take the reference beside the code. On combine each rank
compresses its experts' weighted sum (a folded student's experts already wrote codes through
W_out,i), DeepEP sums the codes on the token's owner, and the owner decompresses and adds its
correction term. The LSH-MoE baseline (`LshAdapter`) sends full rows, has its experts read
cluster centroids instead, and adds each row's scaled offset from its centroid before the combine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from megatron.core.transformer.moe.fused_a2a import get_buffer

from echomoe.lsh import compensate, node_rows, rdma_volume, received_centroids, residual_weights


@dataclass
class Attachment:
    """Every layer's adapter, and the references published in the current forward, each kept
    until the last layer that reads it has decompressed (recompute republishes it inside the
    recomputed chunk)."""

    group: Any = None  # the expert-parallel group the references are gathered over
    adapters: dict[int, Adapter] = field(default_factory=dict)
    last_consumer: dict[int, int] = field(default_factory=dict)  # reference -> last layer using it
    references: dict[int, torch.Tensor] = field(default_factory=dict)  # gathered rows
    own_references: dict[int, torch.Tensor] = field(default_factory=dict)  # this rank's rows
    # A folded student's per-expert readers (`linear_fc1`) and factors (`linear_fc2`), by layer.
    folded_readers: dict[int, Any] = field(default_factory=dict)
    folded_experts: dict[int, Any] = field(default_factory=dict)

    def modules(self) -> dict[int, dict[str, torch.nn.Module]]:
        return {layer: adapter.modules for layer, adapter in self.adapters.items()}

    def store_reference(self, layer: int, rows: torch.Tensor) -> None:
        self.references[layer] = _GatherReference.apply(rows.contiguous(), self.group)
        self.own_references[layer] = rows

    def release_references(self, layer: int) -> None:
        released = [reference for reference, last in self.last_consumer.items() if last == layer]
        for reference in released:
            del self.references[reference]
            del self.own_references[reference]

    def reset_references(self) -> None:
        self.references.clear()
        self.own_references.clear()


class Adapter:
    """The codec around one layer's Flex+DeepEP dispatcher."""

    def __init__(
        self,
        dispatcher: Any,
        modules: dict[str, torch.nn.Module],
        *,
        layer: int,
        dispatch_role: str,
        reference_layer: int | None,
        attachment: Attachment,
        ep_rank: int,
        hidden_size: int,
        folded: bool,
    ) -> None:
        self.dispatcher = dispatcher
        self.modules = modules
        self.dispatch = modules.get("dispatch")
        self.combine = modules.get("combine")
        self.reference_gain = modules.get("dispatch_reference")
        self.correction = modules.get("combine_correction")
        self.layer = layer
        self.dispatch_role = dispatch_role  # off, plain, reference or consumer
        self.reference_layer = reference_layer
        self.attachment = attachment
        self.hidden_size = hidden_size
        self.folded = folded
        self.ep_rank = ep_rank
        self.moe_input = self.route_weights = None

    def install(self) -> None:
        """Wrap the dispatcher's pre- and post-processing seams on both directions."""
        dispatcher = self.dispatcher
        self._dispatch_preprocess = dispatcher.dispatch_preprocess
        self._dispatch_postprocess = dispatcher.dispatch_postprocess
        self._combine_preprocess = dispatcher.combine_preprocess
        self._combine_postprocess = dispatcher.combine_postprocess
        dispatcher.dispatch_preprocess = self.dispatch_preprocess
        dispatcher.dispatch_postprocess = self.dispatch_postprocess
        dispatcher.combine_preprocess = self.combine_preprocess
        dispatcher.combine_postprocess = self.combine_postprocess
        manager = dispatcher._comm_manager
        manager._indices_to_multihot = lambda indices, probs: indices_to_multihot(
            indices, probs, manager.num_local_experts
        )

    def _width(self) -> tuple[int, int]:
        if self.dispatch_role == "reference":
            dispatch = 0  # the experts read the published reference, so the row carries nothing
        elif self.dispatch_role == "off":
            dispatch = self.hidden_size
        else:
            dispatch = self.dispatch.rank
        return dispatch, self.hidden_size if self.combine is None else self.combine.rank

    # --- dispatch -------------------------------------------------------------------------------

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        self.moe_input = hidden_states.reshape(-1, hidden_states.shape[-1])
        routed = routing_map.reshape(-1, routing_map.shape[-1])
        self.route_weights = probs.reshape(-1, probs.shape[-1]).masked_fill(~routed, 0.0)
        # Size DeepEP's buffer for the wider of the two directions now, so the combine never
        # reallocates it under the dispatch's handle.
        get_buffer(
            self.dispatcher._comm_manager.group, max(self._width()) * hidden_states.element_size()
        )
        # the first MoE layer opens a new forward
        if self.layer == min(self.attachment.adapters):
            self.attachment.reset_references()
        prepared, outgoing_probs = self._dispatch_preprocess(hidden_states, routing_map, probs)
        # padding positions (probability zero) route nowhere
        manager = self.dispatcher._comm_manager
        manager.token_indices = manager.token_indices.masked_fill(manager.token_probs == 0, -1)
        return self._encode(prepared), outgoing_probs

    def _encode(self, prepared: torch.Tensor) -> torch.Tensor:
        if self.dispatch_role == "off":
            return prepared
        if self.dispatch_role == "reference":
            self.attachment.store_reference(self.layer, self.moe_input)
            return prepared.new_zeros((prepared.shape[0], 0))
        if self.dispatch_role == "plain":
            return self.dispatch.encode(prepared)
        # non-reference layer: send the code of what the sender's gained reference fails to predict
        staleness = torch.full(
            (prepared.shape[0],),
            self.layer - self.reference_layer,
            dtype=torch.int64,
            device=prepared.device,
        )
        own_reference = self.attachment.own_references[self.reference_layer]
        return self.dispatch.encode(
            prepared - self.reference_gain.encoder_gain(staleness) * own_reference
        )

    def dispatch_postprocess(self, hidden_states, probs):
        if hidden_states.requires_grad:
            hidden_states.register_hook(_contiguous_gradient)
        result = self._dispatch_postprocess(self._decode(hidden_states), probs)
        self.attachment.release_references(self.layer)
        return result

    def _decode(self, received: torch.Tensor) -> torch.Tensor:
        if self.dispatch_role == "off":
            return received
        if self.dispatch_role == "reference":
            return self._reference_rows(received.shape[0], self.layer)
        if self.folded:
            if self.dispatch_role == "plain":
                return received  # a plain layer's `W_in,i = W_i D_d` reads the code itself
            # `W_in,i = W_i [G, D_d]` reads the reference beside the code
            reference = self._reference_rows(received.shape[0], self.reference_layer)
            return torch.cat((reference, received), dim=-1)
        decoded = self.dispatch.decode(received)
        if self.dispatch_role == "plain":
            return decoded
        reference = self._reference_rows(received.shape[0], self.reference_layer)
        staleness = torch.full(
            (reference.shape[0],),
            self.layer - self.reference_layer,
            dtype=torch.int64,
            device=reference.device,
        )
        return self.reference_gain.decoder_gain(staleness) * reference + decoded

    def _reference_rows(self, received_rows: int, layer: int) -> torch.Tensor:
        # Every rank published the same number of rows, so DeepEP's intranode handle (sender and
        # row there) gives a received token's gathered row `source_rank * rows + source_row`.
        rank_prefix_matrix, _, _, source_row, _, _ = self.dispatcher._comm_manager.handle
        # rank_prefix_matrix[i, r] counts the rows rank r receives from ranks 0..i, and each
        # source's rows arrive as one contiguous block in rank order.
        received_prefix = rank_prefix_matrix[:, self.ep_rank].to(torch.int64)
        per_source = torch.diff(received_prefix, prepend=received_prefix.new_zeros(1))
        source = torch.repeat_interleave(
            torch.arange(per_source.shape[0], device=per_source.device),
            per_source,
            output_size=received_rows,
        )
        token_ids = source * self.moe_input.shape[0] + source_row.to(torch.int64)
        # index_select, not rows[token_ids]: advanced indexing backpropagates through a sorted
        # index_put whose nonzero call synchronizes the host inside every non-reference layer's
        # backward.
        return self.attachment.references[layer].index_select(0, token_ids)

    # --- combine --------------------------------------------------------------------------------

    def combine_preprocess(self, hidden_states):
        if self.folded:
            # The experts wrote `r_c`-wide codes, and the manager unpermutes into the width it
            # recorded on dispatch. Name the code width for this one call.
            manager = self.dispatcher._comm_manager
            original = manager.hidden_shape_before_permute
            manager.hidden_shape_before_permute = torch.Size((int(original[0]), self.combine.rank))
            try:
                return self._combine_preprocess(hidden_states)
            finally:
                manager.hidden_shape_before_permute = original
        expert_output = self._combine_preprocess(hidden_states)
        return expert_output if self.combine is None else self.combine.encode(expert_output)

    def combine_postprocess(self, hidden_states):
        try:
            if self.combine is None:
                return self._combine_postprocess(hidden_states)
            decoded = self.combine.decode(hidden_states)
            if self.correction is not None:  # the owner's bias or gated bias decoding term
                decoded = self.correction(
                    self.moe_input, hidden_states, decoded, self.route_weights
                )
            return self._combine_postprocess(decoded)
        finally:
            self.moe_input = self.route_weights = None


class LshAdapter(Adapter):
    """LSH-MoE (`echomoe.lsh`) around one layer's dispatcher, with no codec: the experts read each
    received row's (source rank, local expert, bucket) centroid `c`, and `w alpha_e (x - c)` joins
    their weighted output, with alpha from `scale`. `counts` holds the layer's tokens, DeepEP rows
    and clusters since `lsh_volume` last read them."""

    def __init__(
        self,
        *args: Any,
        rotations: torch.Tensor,
        bucket_count: int,
        scale: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rotations, self.bucket_count, self.scale = rotations, bucket_count, scale
        self.counts = torch.zeros(3, dtype=torch.float64, device=rotations.device)
        self.offsets = self.weights = None

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        result = super().dispatch_preprocess(hidden_states, routing_map, probs)
        routed = self.route_weights > 0  # padding positions route nowhere
        self.counts[0] += routed.any(dim=-1).sum()
        self.counts[1] += node_rows(routed)
        return result

    def dispatch_postprocess(self, hidden_states, probs):
        rows, tokens_per_expert, permuted_probs = super().dispatch_postprocess(
            hidden_states, probs
        )
        manager = self.dispatcher._comm_manager
        centroids, expert, clusters = received_centroids(
            rows,
            hidden_states,
            manager.dispatched_routing_map,
            manager.handle[0],
            self.ep_rank,
            self.rotations,
            self.bucket_count,
        )
        self.counts[2] += clusters
        self.offsets = rows - centroids
        self.weights = residual_weights(permuted_probs, self.scale.alpha, expert)
        return centroids, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states):
        hidden_states = compensate(hidden_states, self.offsets, self.weights)
        self.offsets = self.weights = None
        return super().combine_preprocess(hidden_states)


def lsh_volume(attachment: Attachment, prefix: str) -> dict[str, float]:
    """The RDMA volume of every rank's forwards since the last call, under `prefix`. Every rank
    calls it. Full recompute counts a training forward twice, which leaves each ratio unchanged."""
    adapters = attachment.adapters.values()
    counts = torch.stack([adapter.counts for adapter in adapters])
    torch.distributed.all_reduce(counts)
    for adapter in adapters:
        adapter.counts.zero_()
    return {f"{prefix}/{name}": value for name, value in rdma_volume(counts.tolist()).items()}


# --- what the adapter calls ----------------------------------------------------------------------


def indices_to_multihot(
    indices: torch.Tensor, probs: torch.Tensor, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The manager's multihot conversion by scatter. Megatron's boolean-mask indexing gives the
    same map and probabilities, but its nonzero call stalls the host in every layer."""
    padding_slot = num_local_experts  # an unrouted entry (-1) lands here and is cut off
    slots = indices.masked_fill(indices == -1, padding_slot).to(dtype=torch.int64)
    rows = indices.shape[0]
    routing_map = torch.zeros(rows, padding_slot + 1, dtype=torch.int64, device=indices.device)
    routing_map.scatter_(1, slots, 1)
    multihot_probs = torch.zeros(
        rows, padding_slot + 1, dtype=torch.float, device=probs.device
    ).scatter(1, slots, probs.to(dtype=torch.float))
    return routing_map[:, :padding_slot].bool(), multihot_probs[:, :padding_slot]


def _contiguous_gradient(gradient: torch.Tensor | None) -> torch.Tensor | None:
    # Megatron's `FusedDispatch.backward` records the event DeepEP's stream waits on before it
    # calls `grad_output.contiguous()`. A strided gradient (a folded layer's `torch.cat` hands back
    # one) is then copied after the event and DeepEP's combine reads unwritten rows. Every folded
    # full-model adaptation run was non-finite at update 1 without this. A reference layer's
    # received rows are never read, and their hook gets `None`.
    if gradient is None:
        return None
    return gradient.contiguous()


class _GatherReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, rows: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        world_size = torch.distributed.get_world_size(group)
        gathered = rows.new_empty((world_size * rows.shape[0], *rows.shape[1:]))
        torch.distributed.all_gather_into_tensor(gathered, rows, group=group)
        return gathered

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        world_size = torch.distributed.get_world_size(ctx.group)
        own_gradient = gradient.new_empty((gradient.shape[0] // world_size, *gradient.shape[1:]))
        torch.distributed.reduce_scatter_tensor(
            own_gradient, gradient.contiguous(), op=torch.distributed.ReduceOp.SUM, group=ctx.group
        )
        return own_gradient, None
