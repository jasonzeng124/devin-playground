#!/usr/bin/env bash
# Append one line of GPU and node readings to <out.csv> every <interval> seconds while <pid> lives.
#
#   bash scripts/research/node_trace.sh <out.csv> [interval_s=30] [pid=parent]
#
# Why: r101's native run on a B200 ran a third faster from one minute to the next with nothing changed inside
# the run (LOG 2026-11-19), at the time other jobs on the node ended. A time-budgeted run's step count, and so
# its score, moves with whatever that was. This records what a job can see of it, to line up with the run's
# train log by time. Observation only: it reads this job's GPU (the first index in CUDA_VISIBLE_DEVICES, else
# GPU 0; inside a Slurm job that is the allocated one) and node-wide numbers that name nobody (load, CPU clock,
# pressure). It reads no other process and no other user's job.
#
# Lines: "# ..." (host, GPU, interval), a header starting "utc," (again after a restart), then rows:
#   utc                       time of the reading
#   <nvidia-smi fields>       utilization.gpu (%), utilization.memory (%), power.draw (W), temperature.gpu (C),
#                             memory.used (MiB), then whichever clock and throttle fields this driver knows
#   load1, running            the node's 1-min load average and its runnable tasks
#   mhz_mine, mhz_all         mean "cpu MHz" over this job's CPUs and over all of the node's
#   spin_ms                   wall time of a fixed shell loop on one of this job's CPUs (single-core speed)
#   psi_cpu, psi_mem, psi_io  the node's "some avg60" pressure (%), where the kernel has it
#   cg_cpu_s                  CPU seconds this job's cgroup has used so far
# A reading that is not available is left empty. The seed directory is never created here: rows wait until the
# runner has made it. run_item.sh starts this detached for items whose directory holds a `node_trace` file.
set -u
out=${1:?usage: node_trace.sh <out.csv> [interval_s] [pid]}
every=${2:-30}
pid=${3:-$PPID}
max=${NODE_TRACE_MAX_ROWS:-5000}
gpu=${CUDA_VISIBLE_DEVICES:-}
gpu=${gpu%%,*}
[[ $gpu =~ ^[0-9]+$ ]] || gpu=0

smi() { timeout 20 nvidia-smi -i "$gpu" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed 's/, */,/g'; }

# The base fields are in every driver; the rest are kept only if this one answers them.
fields=utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used
for f in clocks.current.graphics clocks_throttle_reasons.active clocks_event_reasons.active \
         clocks_throttle_reasons.sw_power_cap clocks_throttle_reasons.hw_slowdown \
         clocks_throttle_reasons.sw_thermal_slowdown enforced.power.limit pstate; do
  v=$(smi "$f") || v=
  [[ -n $v && $v != *valid* && $v != *rror* && $v != *,* ]] && fields+=,$f
done
nf=$(awk -F, '{print NF}' <<<"$fields")
blank=$(printf ',%.0s' $(seq 2 "$nf"))

allowed=$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status 2>/dev/null)
cg2=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)/cpu.stat
cg1=/sys/fs/cgroup/cpuacct$(awk -F: '$2 ~ /cpuacct/ {print $3}' /proc/self/cgroup 2>/dev/null)/cpuacct.usage
psi() { awk '/^some/ {sub("avg60=", "", $3); print $3}' "/proc/pressure/$1" 2>/dev/null; }

started=0
n=0
while kill -0 "$pid" 2>/dev/null && (( n < max )); do
  n=$((n + 1))
  if [[ -d $(dirname "$out") ]]; then
    if (( ! started )); then
      {
        echo "# node_trace on $(hostname), GPU $gpu ($(smi name)), every $every s, CPUs $allowed"
        echo "utc,$fields,load1,running,mhz_mine,mhz_all,spin_ms,psi_cpu,psi_mem,psi_io,cg_cpu_s"
      } >>"$out"
      started=1
    fi
    g=$(smi "$fields") || g=
    [[ $(awk -F, '{print NF}' <<<"$g") == "$nf" ]] || g=$blank
    read -r load1 _ _ running _ </proc/loadavg || { load1=; running=; }
    mhz=$(awk -v list="$allowed" '
      BEGIN { k = split(list, a, ","); for (i = 1; i <= k; i++) { m = split(a[i], b, "-"); hi = (m > 1 ? b[2] : b[1]); for (c = b[1]; c <= hi; c++) mine[c + 0] = 1 } }
      /^processor/ { p = $3 + 0 }
      /^cpu MHz/ { all += $4; na++; if (p in mine) { s += $4; ns++ } }
      END { printf "%s,%s", (ns ? sprintf("%.0f", s / ns) : ""), (na ? sprintf("%.0f", all / na) : "") }' /proc/cpuinfo 2>/dev/null)
    t0=$(date +%s%N); i=0; while (( i < 50000 )); do i=$((i + 1)); done; t1=$(date +%s%N)
    spin=$(( (t1 - t0) / 1000000 ))
    cg=$(awk '/^usage_usec/ {printf "%.1f", $2 / 1e6}' "$cg2" 2>/dev/null)
    [[ -z $cg ]] && cg=$(awk '{printf "%.1f", $1 / 1e9}' "$cg1" 2>/dev/null)
    echo "$(date -u +%FT%TZ),$g,$load1,${running%%/*},${mhz:-,},$spin,$(psi cpu),$(psi memory),$(psi io),$cg" >>"$out"
  fi
  sleep "$every"
done
