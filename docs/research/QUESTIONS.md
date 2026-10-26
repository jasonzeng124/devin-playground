# Open questions for the maintainer

## 2026-10-20 18:25 EDT: AWS for dev screens while Engaging is congested?

**Answered 2026-10-21 (laptop, INBOX): approved for a01_skip; seeds 0-2 run on AWS, seeds 3-4 stay
on Engaging.**

Since the maintenance ended (13:45), our workers have barely run. The four that started at 13:46
were preempted after 15-17 min; one restarted at 14:07 and was preempted after 6 min. Nothing has
run since. 2,652 jobs are pending on
mit_preemptable, and Slurm's estimate for our next start is ~01:55 tonight. The two
mit_normal_gpu slots can't start before 2026-10-23.

The AWS approval covers record attempts. Would you also approve dev-ladder screens there for the
strongest candidate, so its promotion decision doesn't wait on Engaging?

- Candidate: `calibration/research_r2/a01_skip` (A 001 + `skip_zero_advantage`). A 001's healthy
  seeds were still rising fast when its budget ended (seed 3: 0.188 -> 0.268 and seed 4: 0.248 ->
  0.300 over the last 50-80 min; LOG 2026-10-20 night). From A 001's step times, skip should fit
  1.6-2.1x more steps into the same budget (estimated; r2 measures it).
- Size: 3 seeds (the 3-seed promotion form) at about 5.3 h each on one H100 80GB HBM3, so about 16
  instance-hours; 5 seeds, about 27.
- Speed-up: on Engaging these 3 seeds are now first in `queue/main.txt` and should finish around
  12:30 on 10-21, if the workers start as estimated and nobody preempts them.
- If yes: say in INBOX which seeds AWS runs, or delete those lines from `queue/main.txt` under
  `flock queue/lock`. Otherwise Engaging would run them too, and I'd have two runs of one seed.

If not, everything stays on Engaging, which is fine.

## 2026-10-21 00:45 EDT: AWS for the track D promotion screen too?

**Closed (laptop, INBOX 2026-10-25): AWS found no H100 capacity in 48 h (on-demand or spot) and
nothing ran there; records don't count on AWS. Earlier: on hold (laptop, INBOX 2026-10-22) until
AWS had shown it can get capacity. Meanwhile the D candidate changed: fp32 lr 3e-6 alone ended at a
0.138 mean (no promotion); the open D screens are fp32 + skip on D 000's and the D2 pretraining
(LOG 2026-10-22 night).**

Engaging still runs nothing (last GPU minute 14:13 on 10-20). 3,071 jobs are pending on
mit_preemptable (2,925 at 22:25), and Slurm now gives our workers no start estimate at all.

The D candidate is fp32 master weights at lr 3e-6 on D 000's recipe. Its seed 0 scored 0.147
against D 000's 0.087 on the same pretrained seed. Promotion needs a 5-seed dev mean >= 0.149
with 4 of 5 seeds above 0.119, so seeds 1-4 decide it. On Engaging they wait behind the stalled
workers.

- Proposal: seeds 1-4 on AWS as dev-only screens of the full pipeline, with no new launcher code:
  `V3=1 STAGE=train RECORD=calibration/research_r2/d00_fp32_lr3e-6_full bash scripts/aws/spot_launch.sh 1 2 3 4`.
  This config (committed) is D 000's record config with `lr` 3e-6 and `master_fp32`, so each
  seed pretrains as D 000 did (112 min) and then runs the fp32 RL (234 min): the record recipe
  minus the test ladder. About 6 h per seed plus D data prep on each instance, so roughly 25
  instance-hours.
- The 5-seed check would count r1's seed 0 (the same RL, from D 000's seed-0 pretraining) with
  these four.
- If yes, say so in INBOX. I then take d00_fp32_lr3e-6 seeds 1-4 out of the Engaging queues
  (seeds 3-4 from `r1`; seeds 1-2 are in `queue/running/`), so nothing runs twice. If Engaging has
  started some of them by then, STATUS.md says which, and AWS can skip those.
- Related: if a01_skip's AWS seeds 0-2 miss the 3-seed form but stay close, the 5-seed form needs
  seeds 3-4. They are near the front of `main` but wait on the same workers. May they also go to
  AWS (about 11 instance-hours)?

If not, everything stays on Engaging.

## 2026-10-22 23:55 EDT: what counts as a "collapsed" seed for promotion?

The promotion rule (PLAN.md) allows at most 1 collapsed seed of 5, and none in the 3-seed form.
`summarize_screens.py` labels a seed one-guess when fewer than 10 of 64 groups are informative.
Read literally, that label blocks every track D arm, because D 000's own seeds are all one-guess.
It also blocks strong track A seeds: `a01_fp32_lr1e-6_skip_cos` seed 0 ended at dev 0.300 with
237-token searches but only ~9 informative groups, because it solves most easy prompts 8 of 8 times.

Clarification applied from now on (PLAN.md, written before seeds 1-2 of that arm or any r4 seed came
in): collapsed = diverged, or one-guess with a final dev below the holder's dev mean. If you want
the literal reading instead, say so in INBOX and I will apply it.

## 2026-10-23 07:00 EDT: is a constrained seed "diverged" when its searches run out of tokens?

`summarize_screens.py` labels a seed diverged when more than half its outputs are malformed, in
training or in the last dev eval ("the policy emits garbage"). Under answer-constrained decoding
(every track A arm since A 001), every answer is well-formed by construction. So "malformed" there
only counts completions that hit the 1,024-token limit before closing the answer, and the dev
score already counts those as failures.

Arms that reward a truncated completion 0 instead of -0.1 (`malformed_reward: 0.0`) learn long
searches. `c90_mal0` ended its 90 min at 0.322 and 0.383 with 62 % and 50 % of greedy dev outputs
truncated, so both seeds are labeled diverged. r5 (`a01_fp32_lr1e-6_skip_cos_mal0`) runs the same
recipe for 290 min; seed 0 was at 0.380 after 117 min with 59 % truncated. Read literally, the label
blocks any such arm from promotion however high it scores.

Reading applied from now on (PLAN.md, written before any r5 seed ended): on constrained arms, a
diverged seed counts as collapsed only when its final dev is below the holder's dev mean, as for
one-guess seeds since 10-22. The record's sampler may then add a think budget (`--think-budget`,
which forces the answer open after N tokens, so nothing truncates), chosen on dev as RULES_v3
requires, and the test ladder decides as for any record. If you want the literal reading, say so in
INBOX: r5 would then not promote, and I would screen penalties that keep truncation under half.

## 2026-10-23 19:35 EDT: should promotion compare dev under the record sampler instead of greedy?

The promotion rule compares in-loop final dev, which is greedy decoding on the 600 dev puzzles. A
record is scored with its own sampler (16 samples at T 0.5, constrained here), and that gains more
over greedy on arms with long searches:

| weights | greedy in-loop | T 0.5, 16 samples | gain |
|---|---|---|---|
| A 001 record seeds (mean of 5) | 0.240 | 0.255 | +0.015 |
| r7 `a01_fp32_lr1e-6_skip_cos_mal0_s300` seed 1 | 0.348 | 0.375 | +0.027 |
| r7, all five seeds (added 22:50) | 0.324 | 0.337 | +0.013 |
| r6 `a01_fp32_lr1e-6_skip_cos_tb768` seed 0 | 0.318 (with r6's think budget 768) | 0.361 | +0.043 |

So the rule may hold back arms that would do well as records, and it can't rank them against arms
that didn't save weights (skip + cosine didn't). Applied: nothing. The rule stays on greedy
in-loop dev, as for every arm so far. If you prefer dev under the record sampler, say so in INBOX.
Arms would then save weights and get a dev eval at T 0.5. The holder line would be A 001's 0.255 +
0.03 = 0.285, and A 002's line would come from its record seeds' evals.

Update 22:50: over all five r7 seeds the gain is +0.013, about A 001's, so seed 1 was the high end.
r6's +0.043 is mostly its in-loop eval forcing answers at 768 tokens. The question matters less than
it first looked. r7 qualifies under either rule (record-sampler mean 0.337, all five seeds above
0.255).

## 2026-10-24 16:35 EDT: may I add node2801 to my pending workers' exclude list?

**Answered 2026-10-25 (laptop, INBOX): yes, and I may update or cancel my own pending jobs without
asking from now on. By then 26649458 and 26649459 had exited (node2801, nothing used up), and I
cancelled 26649456 (pending, no item in hand) to make room for A 002's record workers.**

One of node2801's GPUs is listed by nvidia-smi, but CUDA can't open it ("CUDA-capable device(s)
is/are busy or unavailable", cudaErrorDevicesUnavailable). When it fails a job, Slurm gives the
freed GPU to the next job. Between 15:05 and 16:31, four of my workers landed on it. Each failed
everything it touched within seconds and exited:

- 26644471: D 002 seed 1's second attempt, then r9 seed 1, r10 seed 1 and dm51 seed 0.
- 26644474: r9 seed 0.
- 26644475: r8 seed 1.
- 26649457: dm51 seed 0 again.

Other users' jobs run on the node's other GPUs. What I've done:

- Requeued the failed items.
- Submitted three replacement workers (26786843, 26786844, 26786872) with
  `--exclude=node1708,node1709,node4508,node1800,node2801`.
- Since 16:32, `scripts/research/run_item.sh` tries CUDA before each item. If CUDA fails, it puts
  the item back at the front of the queue. A worker that lands on the GPU then exits without using
  anything up.

**Question.** The successor workers 26649456, 26649458 and 26649459 are pending (each `afterany` on
one of my running workers) and were submitted without node2801 in their exclude list. INBOX
(10-22) says to ask before any `scontrol update` of a pending job. What I would run, for each of
the three:

    scontrol update JobId=<id> ExcNodeList=node1708,node1709,node1800,node2801,node4508

Applied: nothing. If one of them lands on node2801, it exits within a minute without using up an
item, and that GPU slot sits idle until I submit another worker. Answer in INBOX.

## 2026-10-25 02:28 EDT: may track A's H200 record seeds run on mit_preemptable?

**Answered 2026-10-25 11:15 (laptop, INBOX): yes, keeping the mit_normal_gpu chains, with all
workers sharing the record queues. A preempted seed is an infrastructure stop: its partial attempt
goes to extra/ with a note and the same seed reruns. Applied at 11:46 (LOG).**

AGENT.md and INBOX (10-25) put track A records on mit_normal_gpu. That's where they are queued:
two chains of five 6-h jobs, `--gres=gpu:h200:1`, REQUIRE_GPU="NVIDIA H200" (LOG 01:24).

Estimates at 02:24:
- My two chain heads (26824033, 26824038): Slurm plans 10-28 07:00 and 07:30 (reason Priority).
- A new 6-h mit_normal_gpu H200 job (`sbatch --test-only`): 10-27 02:10.
- A new mit_preemptable H200 job with `--time-min=06:30:00`: 10-25 21:48.

Each later round of seeds on mit_normal_gpu waits its turn again. At about one round a day,
A 004 and A 002 would finish around 11-01. On mit_preemptable, a preempted seed is an
infrastructure stop. RULES_v3 lets it rerun under the same seed with the stopped attempt in
`extra/`, as D 002's seed 1 did. That costs up to 5 h of GPU time per preemption. Three of my
four preemptable workers were preempted between 23:40 and 23:51 last night.

**Question.** May track A's H200 record seeds also run on mit_preemptable H200 workers (NVL nodes
excluded, REQUIRE_GPU="NVIDIA H200")? All of a record's seeds would still be on Engaging, the one
provider. Unless you say so, I keep them on mit_normal_gpu as queued. Answer in INBOX.

## 2026-10-26 07:05 EDT: D 003's seeds run 124-127 s over track D's 6-h budget (slow start-up on the nodes); may records use an untimed warm-up?

D 003 seeds 0 and 2 finished with clock `total_s` 21727 and 21724, over the 21600-s budget
(`within_budget: false`). `validate_record_v3.py` rejects an over-budget seed, and dropping a seed
disqualifies a record, so D 003 can't pass as it stands.

Where the time goes: the trainers' own clocks are fixed (pretraining 112 min + GRPO 14040 s =
20760 s). The rest is process start-up (imports, tokenizer, model creation, vLLM initialisation)
and the final saves, all inside `run_seed_v3.sh`'s timed stages:

| seeds | GPU | pretraining over its 112 min | GRPO process over its training clock | total |
|---|---|---|---|---|
| D 002, 5 seeds | H100 | 9-13 s | 114-197 s | 20884-20971 |
| D 003 seed 0 | H200 (node4900, CPUs oversubscribed) | 156 s | 810 s | 21727 |
| D 003 seed 2 | H200 (node4300) | 486 s | 477 s | 21724 |

It looks like cold reads of the Python environment (`~/rlvr-venv` on NFS) on the compute nodes.
The workers' 150-s CUDA check (`import torch; torch.zeros(1, device="cuda")`) timed out on 15 of
the 42 starts since 10-24 23:00. The same nodes started fast at other times: node4300 at 02:00 and
05:42, and node2901, where r16 seed 0 restarted with vLLM ready in 3.5 min. On node2904, r16
seed 0 took 15 min from launch to vLLM's first log line. The login node imports torch in 3-6 s.

Track A has 600 s of margin (18000 against 17400), and A 001 used 33-98 s on the H100. An A record
seed that starts on a cold node would probably be over too.

What I did (all reversible):
- Held A 004's and A 002's ten items in `queue/held_records.txt`, which no worker reads. Neither
  record has started or scored a seed, so its N and seeds are untouched.
- D 003's seeds 1, 3 and 4 keep running, because stopping them would drop runs.
- The H200 workers run r16 screens in the meantime.

Proposal: an untimed warm-up before each timed stage of `run_seed_v3.sh` (a new option, off by
default). It would read the environment's shared libraries into the node's page cache and import
torch, transformers and vLLM in a process that sees no GPU (`CUDA_VISIBLE_DEVICES=`), before `t0`.
RULES_v3 starts the clock at the first GPU operation, and this does none. The CUDA check (a GPU
call) already runs before `t0` in every record seed. I'll build it and measure it on screens. No
record uses it until you answer.

**Questions.**
1. May record seeds use that warm-up?
2. D 003: may I report it as failed (over budget, all five seeds kept in the record folder with a
   note) and rerun D 002's config, unchanged, with the warm-up as D 004? Or would you rather
   shorten the reruns' budgets, for example pretraining 108 min, to leave margin?

Answer in INBOX.
