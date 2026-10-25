# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. The H200 board starts with three record runs (LOG 10-25 01:24):**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | preemptable H200 workers | seed 0 done (test 0.1590, dev 0.168); seed 1 reruns when 26824150 restarts; seeds 2-4 queued |
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | preemptable H200 workers and mit_normal_gpu chain 1 (26824033-37) | seeds 0-4 queued |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | preemptable H200 workers and mit_normal_gpu chain 2 (26824038-42) | seeds 0-4 queued |

- A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and the same seed
  reruns (INBOX 10-25 11:15).
- A 004 and A 002 run side by side; A 002 is compared with A 004 once both land. A 003
  (`records/v3/track_a/003_fp32_mal0_s300`) waits, as the laptop asked.
- Promotion uses the H100 lines until the reruns land: A 0.2703 (163 of 600 per seed for the
  3-seed form), D 0.190 (115 of 600). Then the lines come from the reruns' in-loop final dev (PLAN).
- node4900 is excluded from every worker: its CPUs are oversubscribed, and D's RL ran at half speed
  there (LOG 14:30).

H100 board (closed 10-24): A 001 holds track A (test 0.2481, dev 0.2403); D 002 holds track D
(test 0.1512 ± 0.0202, dev 0.160).

Promoted on H100 dev numbers, not yet recorded: A 002 (dev 0.324; seeds 0.300 / 0.270 / 0.472 /
0.315 / 0.263) and A 003 (dev 0.324; seeds 0.385 / 0.348 / 0.238 / 0.348 / 0.302; about one seed in
five of its recipe collapses between steps 250 and 300).

Updated 2026-10-25 19:10 EDT.

## AWS box 1: 8x B200, 2026-10-26 11:30Z (07:30 EDT) to 10-27 11:30Z

Dev-ladder screens only (no records, no test-ladder scoring), two runs per GPU, step-capped (LOG 14:30).
- Queue `queue/b200.txt`: r15, seeds 0-3 of eight arms (32 items, ~5-7 h each at two per GPU; `mal0_st500`
  replaced `advnone_st800` at 18:46, ~3.5-4 h; `mal0_st800_tis2` samples ~2x slower, LOG 18:46).
- Start (the box launches itself at +1/+10/+25 min; launch it by hand if nothing runs at +30 min:
  `aws ec2 run-instances --region us-east-1 --launch-template LaunchTemplateName=rlvr-box-cr-03fec7969fbda8633`).
  `B="env BOX=zjason@100.29.118.132 bash scripts/aws/box_ssh.sh"` (BatchMode; host key in `agent/known_hosts_<ip>`).
  1. `$B 'cat /var/lib/rlvr-box.ready; tail -3 /var/log/rlvr-box.log; ls -d /usr/local/cuda*; /usr/local/cuda/bin/nvcc --version | tail -2; nvidia-smi -L'`.
     vLLM's cu130 wheels JIT-compile with nvcc: if `/usr/local/cuda` is older than 13.0, set CUDA_HOME to a 13.x
     toolkit on the box (box_workers.sh keeps a CUDA_HOME it is given).
  2. Copy (~10.6 GB) as a mit_quicktest job (mit_normal waits days; compute nodes egress as 18.13.54.x):
     `sbatch -p mit_quicktest -c 4 --mem=8G --time=00:15:00 -o slurm_logs/probe/box1_sync_%j.out --wrap 'BOX=zjason@100.29.118.132 bash scripts/aws/box_sync.sh to b200 orcd/pool/rlvr-ckpts/calibration/research_r7/a01_fp32_lr1e-6_skip_cos_mal0_s300/seeds/1/final orcd/pool/rlvr-ckpts/records/v3/track_d/002_dm_math_fp32/seeds/0/pretrained'`
     (rsync resumes if it needs a second job).
  3. Staged start (`W="QUEUE=b200 PER_GPU=2 RUNNER=scripts/research/run_item.sh RESERVE_MIN=420"`):
     - the probe on GPU 7, then GPU 7's workers when it ends:
       `$B 'cd ~/rlvr-screens && export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH && setsid nohup bash -c "CUDA_VISIBLE_DEVICES=7 MEM_1=0.2 MEM_2=0.2 bash scripts/research/pack_probe.sh >slurm_logs/probe.out 2>&1; GPUS=7 QUEUE=b200 PER_GPU=2 RUNNER=scripts/research/run_item.sh RESERVE_MIN=420 bash scripts/aws/box_workers.sh" </dev/null >/dev/null 2>&1 &'`
     - `$B "cd ~/rlvr-screens && GPUS=0 $W bash scripts/aws/box_workers.sh </dev/null"` (stdin from /dev/null, so ssh returns);
     - once both GPU 0 runs log steps (`tail -2 ~/orcd/pool/rlvr-ckpts/calibration/research_r15/*/seeds/*/train_log.jsonl`;
       `nvidia-smi` for memory): `$B "cd ~/rlvr-screens && GPUS='1 2 3 4 5 6' $W bash scripts/aws/box_workers.sh </dev/null"`.
  4. Every wake: `BOX=zjason@100.29.118.132 bash scripts/aws/box_sync.sh from box1 calibration/research_r15`, then
     `summarize_screens.py calibration/research_r15/*`; commit the seeds' logs. The probe's output is
     `slurm_logs/probe.out` on the box (copy it by hand).
- The tail of the block: workers stop taking 800-step items 7 h before the deadline (block end - 35
  min); shorter items for the last hours get decided from the first results.

## Running

- mit_preemptable H100 worker (`main:long`) 26803673 runs r14 seed 1 on node2700 since 18:22 (ends
  ~23:20), then r16 seed 0 (`main`, ~23:20 to ~03:30). 26786843 finished r10 seed 2 and exited
  (18:42). Their H200 successors 26854533 (pending, eligible) and 26854534 (`afterany` 26803673) read
  `record-003`, both A record queues, then `h200`.
- H200 workers on mit_preemptable (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900
  excluded): 26824150 (requeued at 12:10, pending; reruns D 003 seed 1 first), 26862717 (submitted 14:22 to replace node4900's 26824148; reads `record-003`, A 004's and A 002's
  queues, then `h200`). Slurm estimates new H200 starts at 10-26 ~07:45.
- mit_normal_gpu: the two record chains, pending (Slurm's plan: 10-27).
- Cap: 2 normal + 4 preemptable = 6. I may update or cancel my own pending jobs without asking.

## Queues

- `record-003_dm_math_fp32_h200`: seeds 2-4. `record-004_ladder_constrained_h200`,
  `record-002_fp32_skip_cosine`: seeds 2-4 each.
- `h200` (4): A 004 seed 0, A 002 seed 0, A 004 seed 1, A 002 seed 1 (r12 left at 18:46: r10 closed).
- `main` (1): r16 `mal0_st500` seed 0. `b200` (32): box 1. `h100`, `later`: parked, read by no worker.

## Best so far

- Track A, H100 dev: A 002 and A 003 (0.324). Best single screen seed: r14 seed 0 (r5 + TIS) 0.398
  (best 0.410), held by its slow sampling's annealed lr, not by TIS (LOG 18:46); r16 / r15
  `mal0_st500` test that. Closed: r10 (mean-centred advantages, 0.261 on three seeds), r9 (Clip-Cov),
  r13 (KL 0.05 with mean-centred advantages), W-REINFORCE, training temperature 1.2 in the -0.1 regime.
- Track D, H100 dev: D 002's recipe 0.153 (screens) / 0.160 (record seeds). Closed: dm51 (+0.010 over
  dm26 on three seeds), r11 (lr 1e-5, 3e-5).

## Next

- ~23:20: r14 seed 1 ends (second seed of the slow-clock condition, mismatch logged); r16 seed 0
  starts, the first GPU run of `budget_steps` (check its first steps and `clock_s` before box 1).
- ~03:30: r16 seed 0 ends. If it holds near 0.40 past step 375, `mal0_st500` goes to the front of
  box 1's queue and gets an H200 screen after the A record seeds (a record config would also need a
  hard time stop next to `budget_steps`, a new option).
- 06:30 tomorrow: box 1 prep, then wakes every 20-30 min until the block ends.
- By ~18:00Z tomorrow (first r15 items done): the box's work after r15 (more seeds of the best
  arms, or token-capped track D screens: `pretrain_scratch.py` decays on `--max-tokens` when it
  runs out before `--minutes`).
- Track D H200 screens can start from D 003's saved pretrained models once seeds finish.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q` (two-rank tests fail on the login node otherwise).
- Eval arms: a dir with `eval.json`, queued like a screen, runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`). `summarize_screens.py` prints them.
- Step-capped runs (`budget_steps`) log `clock_s`; `summarize_screens.py` reads their minutes from it.
- Track D pretraining is time-budgeted and its speed varies by node; read D results with their
  tokens/s.
- Ubuntu 24 nodes: the worker adds gcc's multiarch paths; `module load` fails there.
