<!-- Delete the section that does not apply. -->

## Record submission

Record: `records/<track>/<NNN>_<slug>/` — one sentence on what changed vs. the parent record.

| | |
|---|---|
| mean ± sd (N) | `... ± ... s (N=...)` |
| 95% CI | `[..., ...] s` |
| vs. holder | `welch p = ..., permutation p = ...` |
| guardrail | `FineWeb Δ ..., MMLU-lite Δ ...` |
| provider | `...` (all seeds) |

<details><summary><code>validate_record.py</code> output (with <code>--compare-to</code>)</summary>

```
paste here
```
</details>

Checklist (see [RULES.md](../blob/main/RULES.md)):

- [ ] `config.json` declares `track`, `model`, `hardware`, `provider`, `seeds: [0..N-1]` (N ≥ 3) and the RL hyperparameters
- [ ] every declared seed has `seeds/<seed>/{config.json,train_log.jsonl,eval_log.jsonl,result.json,probe.json}`
- [ ] all seeds ran on the one declared provider, on the ranked GPU (`NVIDIA H100 80GB HBM3`)
- [ ] every run launched with this configuration is disclosed (official seeds, `extra/`, or the README)
- [ ] no model weights committed; nothing touched `data/`
- [ ] added a row to the leaderboard table in `README.md`
- [ ] `python scripts/validate_all_records.py` passes locally

## Code change

What and why. If it changes timing, generation, the verifier or the eval path,
say how you checked that existing records are unaffected (or why they cannot be).
