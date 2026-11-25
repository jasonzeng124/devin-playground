"""CPU tests of scripts/pretrain_scratch.py's data-parallel pieces (--ddp) and its resume file (--resume)."""

from pathlib import Path

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp

from scripts.pretrain_scratch import accumulate, allreduce_grads, load_resume, save_resume, step_layout

WORLD = 2
# (shape, dtype): the dtype change and the 64-byte buckets make several buckets, one parameter has no grad.
SHAPES = [((3, 4), torch.float32), ((5,), torch.float32), ((2, 2), torch.float64), ((7,), torch.float32), ((1,), torch.float32)]


def _grads(rank: int) -> list:
    g = torch.Generator().manual_seed(100 + rank)
    return [torch.randn(shape, generator=g, dtype=dtype) for shape, dtype in SHAPES]


def _allreduce_worker(rank: int, init_file: str, out_dir: str):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    params = [torch.nn.Parameter(torch.zeros(shape, dtype=dtype)) for shape, dtype in SHAPES]
    for p, g in zip(params, _grads(rank)):
        p.grad = g
    params[3].grad = None
    allreduce_grads(params, bucket_bytes=64)
    torch.save([None if p.grad is None else p.grad for p in params], Path(out_dir) / f"{rank}.pt")
    torch.distributed.destroy_process_group()


def test_allreduce_grads_averages_over_ranks(tmp_path):
    mp.spawn(_allreduce_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=WORLD, join=True)
    per_rank = [_grads(r) for r in range(WORLD)]
    for rank in range(WORLD):
        got = torch.load(tmp_path / f"{rank}.pt")
        for i, (shape, dtype) in enumerate(SHAPES):
            if i == 3:
                assert got[i] is None
                continue
            assert got[i].dtype == dtype and got[i].shape == shape
            torch.testing.assert_close(got[i], sum(g[i] for g in per_rank) / WORLD)


@pytest.mark.parametrize("batch_tokens,seq,micro,max_tokens", [(524288, 1024, 64, 1_400_000_000), (524288, 1024, 16, 10**9), (1000, 1024, 64, 10**7)])
def test_step_layout_on_one_gpu_is_the_old_layout(batch_tokens, seq, micro, max_tokens):
    accum = max(1, batch_tokens // (seq * micro))
    assert step_layout(batch_tokens, seq, micro, 1, max_tokens) == (accum, seq * micro * accum, max_tokens // (seq * micro * accum))


def test_step_layout_keeps_the_global_batch():
    accum, step_tokens, total = step_layout(524288, 1024, 16, 8, 8 * 10**9)
    assert (accum, step_tokens, total) == (4, 524288, 8 * 10**9 // 524288)


def _model_and_opts(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    hidden = [model[0].weight, model[2].weight]
    rest = [model[0].bias, model[2].bias]
    opts = [torch.optim.Muon(hidden, lr=0.02, weight_decay=0.1, adjust_lr_fn="match_rms_adamw"), torch.optim.AdamW(rest, lr=0.02)]
    return model, opts


def _step(model, opts, x):
    model(x).pow(2).mean().backward()
    for opt in opts:
        opt.step()
        opt.zero_grad(set_to_none=True)


def test_resume_continues_bit_identically(tmp_path):
    xs = [torch.randn(4, 8, generator=torch.Generator().manual_seed(i)) for i in range(5)]
    model, opts = _model_and_opts(0)
    for x in xs[:2]:
        _step(model, opts, x)
    path = tmp_path / "resume" / "state.pt"
    meta = {"step": 2, "tokens": 64, "elapsed": 12.5}
    save_resume(path, model, opts, meta)
    assert path.exists() and not path.with_name("state.pt.tmp").exists()
    for x in xs[2:]:
        _step(model, opts, x)

    resumed, resumed_opts = _model_and_opts(1)  # other weights, fresh optimizers
    assert load_resume(path, resumed, resumed_opts) == meta
    for x in xs[2:]:
        _step(resumed, resumed_opts, x)
    for (name, a), b in zip(model.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(a, b), name


def test_resume_refuses_another_optimizer_count(tmp_path):
    model, opts = _model_and_opts(0)
    save_resume(tmp_path / "state.pt", model, opts, {"step": 0, "tokens": 0, "elapsed": 0.0})
    with pytest.raises(SystemExit):
        load_resume(tmp_path / "state.pt", model, opts[:1])


ACCUM = 3


def _overlap_worker(rank: int, init_file: str, out_dir: str):
    """Two steps of three micro-batches each way: the plain backward + allreduce_grads (--ddp) and DDP with no_sync
    on the first two micro-batches (--ddp-overlap), same start, same data per rank, set_to_none between steps."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    out = {}
    for way in ("plain", "overlap"):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
        net = None
        if way == "overlap":
            net = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True, bucket_cap_mb=0.0002)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        data = torch.Generator().manual_seed(50 + rank)
        fwd = net if net is not None else model

        def draw(data=data):
            return torch.randn(4, 8, generator=data), torch.randint(0, 3, (4,), generator=data)

        def loss_fn(x, y, fwd=fwd):
            return torch.nn.functional.cross_entropy(fwd(x), y)

        grads = []
        for _ in range(2):
            accumulate(loss_fn, draw, ACCUM, net)
            if net is None:
                allreduce_grads(model.parameters())
            grads.append([p.grad.detach().clone() for p in model.parameters()])
            opt.step()
            opt.zero_grad(set_to_none=True)
        out[way] = {"grads": grads, "weights": [p.detach().clone() for p in model.parameters()]}
    torch.save(out, Path(out_dir) / f"{rank}.pt")
    torch.distributed.destroy_process_group()


def test_ddp_overlap_matches_allreduce_grads(tmp_path):
    mp.spawn(_overlap_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=WORLD, join=True)
    got = [torch.load(tmp_path / f"{r}.pt") for r in range(WORLD)]
    for rank in range(WORLD):
        plain, overlap = got[rank]["plain"], got[rank]["overlap"]
        for gp, go in zip(plain["grads"], overlap["grads"]):
            for a, b in zip(gp, go):
                torch.testing.assert_close(b, a)
        for a, b in zip(plain["weights"], overlap["weights"]):
            torch.testing.assert_close(b, a)
        for a, b in zip(got[0]["overlap"]["weights"], overlap["weights"]):
            torch.testing.assert_close(b, a)  # the ranks stay in step
    assert not torch.allclose(got[0]["plain"]["grads"][0][0], torch.zeros(()))
