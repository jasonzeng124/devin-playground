"""CPU (gloo) tests of the data-parallel path: sharded update == single-process update on the whole batch."""

import json
import os
from pathlib import Path

import torch
import torch.distributed
import torch.multiprocessing as mp
from transformers import GPT2Config, GPT2LMHeadModel

import rlvr_speedrun.grpo as grpo
from rlvr_speedrun.distributed import Dist, shard_bounds
from rlvr_speedrun.grpo import GRPOConfig, _update, run
from tests.test_grpo import TinyTokenizer

WORLD = 2


def _tiny_model():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=40, n_positions=64, n_ctx=64, n_embd=16, n_layer=1, n_head=2, pad_token_id=0, eos_token_id=1)
    )
    return model.eval()  # no dropout, as load_causal_model does


def _fixed_batch():
    torch.manual_seed(1)
    rows, prompt_width, completion = 8, 6, 5
    generated = torch.randint(2, 40, (rows, prompt_width + completion))
    attention = torch.ones_like(generated)
    attention[:, prompt_width + 3 :] = torch.tensor([[1, 1], [1, 0], [0, 0], [1, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25])
    return generated, attention, advantages, prompt_width


def _join(rank: int, world_size: int, init_file: Path) -> Dist:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    return Dist(rank=rank, world_size=world_size)


def _update_worker(rank: int, init_file: str, out_file: str):
    dist = _join(rank, WORLD, Path(init_file))
    model = _tiny_model()
    config = GRPOConfig(lr=1e-3, micro_batch_size=3, kl_coef=0.0, grad_clip=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    generated, attention, advantages, prompt_width = _fixed_batch()
    start, end = shard_bounds(generated.shape[0], rank, WORLD)
    _loss, _kl, _entropy, token_count = _update(
        model, None, generated[start:end], attention[start:end], advantages[start:end], prompt_width, 0, config, optimizer, torch.device("cpu"), dist
    )
    if rank == 0:
        torch.save({"params": torch.nn.utils.parameters_to_vector(model.parameters()).detach(), "tokens": int(token_count)}, out_file)
    torch.distributed.destroy_process_group()


def test_sharded_update_matches_single_process(tmp_path):
    mp.spawn(_update_worker, args=(str(tmp_path / "init"), str(tmp_path / "out.pt")), nprocs=WORLD, join=True)
    sharded = torch.load(tmp_path / "out.pt")

    model = _tiny_model()
    config = GRPOConfig(lr=1e-3, micro_batch_size=3, kl_coef=0.0, grad_clip=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    generated, attention, advantages, prompt_width = _fixed_batch()
    _loss, _kl, _entropy, token_count = _update(
        model, None, generated, attention, advantages, prompt_width, 0, config, optimizer, torch.device("cpu"), Dist()
    )
    assert sharded["tokens"] == int(token_count) == int(attention[:, prompt_width:].sum())
    single = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(sharded["params"], single, atol=1e-6, rtol=0)


def _run_worker(rank: int, init_file: str, out_dir: str, params_file: str):
    dist = _join(rank, WORLD, Path(init_file))
    model = _tiny_model()
    grpo.load_tokenizer = lambda _name: TinyTokenizer()
    grpo.load_causal_model = lambda _name, _device, _dtype: (model, torch.device("cpu"))
    config = GRPOConfig(
        model="tiny",
        group_size=2,
        prompts_per_step=4,
        max_new_tokens=4,
        max_steps=3,
        eval_every=1,
        eval_limit=7,
        eval_batch_size=2,
        kl_coef=0.0,
        no_save=True,
        device="cpu",
        out_dir=out_dir,
    )
    result = run(config, dist)
    torch.save(torch.nn.utils.parameters_to_vector(model.parameters()).detach(), f"{params_file}.{rank}.pt")
    Path(f"{params_file}.{rank}.json").write_text(json.dumps(result))
    torch.distributed.destroy_process_group()


def test_run_two_ranks_writes_once_and_keeps_replicas_in_sync(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)  # run() reads data/countdown_eval.jsonl
    out_dir = tmp_path / "ddp"
    mp.spawn(_run_worker, args=(str(tmp_path / "init"), str(out_dir), str(tmp_path / "params")), nprocs=WORLD, join=True)
    params = [torch.load(f"{tmp_path}/params.{r}.pt") for r in range(WORLD)]
    results = [json.loads(Path(f"{tmp_path}/params.{r}.json").read_text()) for r in range(WORLD)]
    assert torch.equal(params[0], params[1])
    assert results[0]["total_steps"] == 3
    assert results[0]["environment"]["world_size"] == WORLD
    assert results[0]["pass_rate"] == results[1]["pass_rate"]
    result = json.loads((out_dir / "result.json").read_text())
    assert result["total_steps"] == 3
    train = [json.loads(line) for line in (out_dir / "train_log.jsonl").read_text().splitlines()]
    assert len(train) == 3 and all(row["kept_groups"] == 4 for row in train)
    evals = [json.loads(line) for line in (out_dir / "eval_log.jsonl").read_text().splitlines()]
    assert len(evals) == 3 and all(row["n"] == 7 for row in evals)


def test_shard_bounds_cover_everything_in_order():
    for n in range(0, 12):
        for world in (1, 2, 3, 8):
            pieces = [shard_bounds(n, r, world) for r in range(world)]
            assert pieces[0][0] == 0 and pieces[-1][1] == n
            assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))
            assert max(e - s for s, e in pieces) - min(e - s for s, e in pieces) <= 1


def test_prompts_per_step_must_divide_by_world_size(tmp_path):
    config = GRPOConfig(model="tiny", prompts_per_step=3, out_dir=str(tmp_path))
    try:
        run(config, Dist(rank=0, world_size=2))
    except ValueError as err:
        assert "divisible" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_single_process_is_default_when_no_torchrun_env(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    from rlvr_speedrun.distributed import init_from_env

    dist, device = init_from_env("cpu")
    assert not dist.enabled and device == "cpu" and os.environ.get("WORLD_SIZE") is None
