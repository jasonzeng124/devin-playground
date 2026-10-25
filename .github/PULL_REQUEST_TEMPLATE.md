<!-- Delete the section that does not apply. -->

## Record submission (benchmark v3)

Record: `records/v3/track_<a|d>/<NNN>_<slug>/`: one sentence on what changed vs. the holder.

| | |
|---|---|
| test ladder, mean ± sd (N) | `0.... ± 0.... (N=...)` |
| vs. holder | `Welch t = ..., one-sided p = ...` |
| sampler | `temperature ..., constrain ..., think_budget ...`, or a custom sampler described in the README |
| clock per seed | `... min` (track D: pretraining + RL) |
| GPU / provider | `NVIDIA H200` / `...` (all seeds) |

<details><summary><code>validate_record_v3.py --compare-to</code> output</summary>

```
paste here
```
</details>

Checklist (see [docs/RULES_v3.md](../blob/main/docs/RULES_v3.md)):

- [ ] >= 5 seeds `0..N-1`, each with `seeds/<i>/{config.json, train_log.jsonl, eval_log.jsonl, result.json, clock.json, confirm_v3.json}` (track D: plus `pretrain_log.jsonl` from that seed's own pretraining)
- [ ] every run launched with this configuration is reported (the seeds, or `extra/` with a note)
- [ ] all seeds trained on one provider, on the ranked GPU, within the track's budget
- [ ] in-loop evals used the dev ladder; the test ladder was scored once per seed, after training
- [ ] training used only generated puzzles, verifier verdicts, the model's own outputs and public datasets that aren't Countdown-specific (each named with its revision in the README); no solutions, no other models, no arithmetic text generated for the submission
- [ ] any custom sampler, reward code or pretraining script is in the record folder and described in its README
- [ ] no model weights committed; nothing touched `data/`
- [ ] added a row to the v3 leaderboard in `README.md`
- [ ] `uv run python scripts/validate_all_records.py` passes locally

## Code change

What and why. If it changes the timed path, rollouts, the verifier, the ladders or the scorer,
say how existing records are affected (or why they cannot be).
