"""Data-parallel helpers for `torchrun` launches of the GRPO trainer.

Every rank holds a full replica of the policy. Each step, the (identical, seeded)
global prompt list is split into contiguous per-rank shards; ranks sample and
score their shard, then gradients are summed across ranks with the loss normalised
by the *global* completion-token count, so the update is exactly the single-process
update on the concatenated batch. Evaluation is sharded the same way and the
completions are gathered back in order. Backend is gloo on CPU (used by the tests)
and nccl on CUDA.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def shard_bounds(n: int, rank: int, world_size: int) -> tuple[int, int]:
    """[start, end) of rank's contiguous shard of n items; the first n % world_size ranks get one extra."""
    base, extra = divmod(n, world_size)
    start = rank * base + min(rank, extra)
    return start, start + base + (1 if rank < extra else 0)


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def shard(self, items: Sequence[T]) -> list[T]:
        start, end = shard_bounds(len(items), self.rank, self.world_size)
        return list(items[start:end])

    def all_reduce_sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather_object(self, obj: T) -> list[T]:
        if not self.enabled:
            return [obj]
        gathered: list[T] = [obj] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def all_reduce_grads_(self, parameters) -> None:
        """Sum gradients across ranks in one flat collective (no-op when not distributed)."""
        if not self.enabled:
            return
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for g in grads:
            g.copy_(flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()


SINGLE = Dist()


def init_from_env(device: str) -> tuple[Dist, str]:
    """Join the process group described by torchrun's env vars; returns (Dist, per-rank device string).

    Without WORLD_SIZE > 1 in the environment this is a no-op returning the
    single-process `Dist()` and the device unchanged.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return Dist(), device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    dist.init_process_group("nccl" if use_cuda else "gloo")
    return Dist(rank=rank, world_size=world_size, local_rank=local_rank), device


def destroy() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
