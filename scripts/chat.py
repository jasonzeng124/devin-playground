"""Talk to saved models from a terminal: the base model, any record checkpoint, or a path.

These are completion models, not chat models: the model continues whatever text it is given.
A line of plain text is continued as it is. `/p` and `/r` wrap a puzzle in the benchmark's prompt,
sample one completion the way the record's sampler does (temperature 1, the answer constraint if
the record used it, stop at `</answer>`), and check it with the verifier.

Run it on a compute node through `scripts/chat.sh` (never on a login node). Names:

    a-base            Qwen/Qwen3-1.7B-Base before any training
    a-006             track A record 006, seed 0, final weights (`a-006-s3` = seed 3)
    d-006-pre         track D record 006, seed 0, after pretraining and before RL
    d-006-s2          track D record 006, seed 2, final weights
    a-collapsed ...   shortcuts listed by `/models`
    <path or HF id>   anything `AutoModelForCausalLM.from_pretrained` loads

Only the dev ladder is read (`/r`); the test ladder is never touched.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rlvr_speedrun import constrained  # noqa: E402
from rlvr_speedrun.countdown import ALL_OPS, Puzzle, format_prompt, verify  # noqa: E402

CKPT = Path(os.environ.get("CKPT_DIR", str(Path.home() / "orcd/pool/rlvr-ckpts")))
BASE = "Qwen/Qwen3-1.7B-Base"
DEV = REPO / "data/ladder_v3_dev.jsonl"
NAME_RE = re.compile(r"^([ad])-(\d{3})(?:-s(\d+))?(-pre)?$")

# shortcut -> (name, what it is); scores are from the record's own eval_log.jsonl and confirm_v3.json
SHORTCUTS = {
    "a-first": ("a-000-s3", "track A baseline (test 0.146)"),
    "a-best": ("a-006-s4", "track A holder, best seed (test 0.428)"),
    "a-collapsed": ("a-005-s3", "track A 005 seed 3: dev peaked 0.335 at step 250, ended 0.197 at step 400"),
    "a-collapsed2": ("a-002-s3", "track A 002 seed 3: dev peaked 0.373 at step 750, ended 0.183 at step 1345"),
    "a-collapsed3": ("a-002-s1", "track A 002 seed 1: dev peaked 0.353 at step 350, ended 0.210 at step 2431"),
    "d-pre": ("d-006-s2-pre", "track D holder's seed 2 after pretraining, before any RL"),
    "d-first": ("d-000-s1", "track D baseline (test 0.139)"),
    "d-best": ("d-006-s2", "track D holder, best seed (test 0.266)"),
}


def record_dir(track: str, number: str) -> Path | None:
    found = sorted((REPO / f"records/v3/track_{track}").glob(f"{number}_*"))
    return found[0] if found else None


def describe(record: Path, seed: str, pre: bool) -> str:
    """One line on a record seed from its committed logs."""
    seed_dir = record / "seeds" / seed
    parts = [f"{record.parent.name[-1].upper()} {record.name}, seed {seed}"]
    if pre:
        return parts[0] + ", after pretraining (before RL)"
    try:
        rows = [json.loads(line) for line in (seed_dir / "eval_log.jsonl").read_text().splitlines() if line.strip()]
        peak, last = max(rows, key=lambda r: r["pass_rate"]), rows[-1]
        parts.append(f"dev peak {peak['pass_rate']:.3f} at step {peak.get('step')}, last {last['pass_rate']:.3f} at step {last.get('step')}")
        if last.get("kept_step") is not None:
            parts.append(f"weights kept from step {last['kept_step']}")
    except (OSError, ValueError, KeyError):
        pass
    try:
        parts.append(f"test {json.loads((seed_dir / 'confirm_v3.json').read_text())['ladder_score']:.3f}")
    except (OSError, ValueError, KeyError):
        pass
    return ", ".join(parts)


def resolve(name: str) -> tuple[str, str, dict]:
    """A name -> (what to load, a one-line description, {"max_new_tokens", "constrain"})."""
    note = ""
    if name in SHORTCUTS:
        name, note = SHORTCUTS[name]
    if name == "a-base":
        return BASE, "Qwen3-1.7B-Base before any training", {"max_new_tokens": 1024, "constrain": True}
    match = NAME_RE.match(name)
    if not match:
        return name, name, {"max_new_tokens": 512, "constrain": False}
    track, number, seed, pre = match.group(1), match.group(2), match.group(3) or "0", bool(match.group(4))
    record = record_dir(track, number)
    if record is None:
        raise ValueError(f"no record {number} in records/v3/track_{track}")
    if pre and track != "d":
        raise ValueError("-pre exists only for track D (track A starts from a-base)")
    config = json.loads((record / "config.json").read_text())
    weights = CKPT / "records/v3" / f"track_{track}" / record.name / "seeds" / seed / ("pretrained" if pre else "final")
    if not weights.is_dir():
        raise ValueError(f"no weights at {weights}")
    settings = {"max_new_tokens": int(config.get("max_new_tokens", 1024)), "constrain": bool(config.get("constrain_answer", False))}
    return str(weights), note or describe(record, seed, pre), settings


def list_models() -> str:
    lines = ["shortcuts:", f"  {'a-base':13} Qwen3-1.7B-Base before any training"]
    lines += [f"  {short:13} = {name}: {note}" for short, (name, note) in SHORTCUTS.items()]
    lines.append("records (name-sN for seed N; track D also -pre for the weights after pretraining):")
    for track in "ad":
        for record in sorted((REPO / f"records/v3/track_{track}").glob("[0-9][0-9][0-9]_*")):
            seeds_dir = CKPT / "records/v3" / f"track_{track}" / record.name / "seeds"
            seeds = sorted(p.name for p in seeds_dir.iterdir()) if seeds_dir.is_dir() else []
            if seeds:
                lines.append(f"  {track}-{record.name[:3]}  seeds {','.join(seeds)}  {record.name}")
    return "\n".join(lines)


class Session:
    def __init__(self, temperature: float, log: str = ""):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
        self.temperature = temperature
        self.log = log  # a JSONL file that gets one line per completion ("" = none)
        self.name = self.about = ""
        self.max_new_tokens: int | None = None  # None = the loaded model's own setting
        self.constrain: bool | None = None
        self.model = self.tokenizer = self.grammar = None
        self.settings: dict = {}
        self.last: tuple[str, Puzzle | None] | None = None
        self.dev: list[Puzzle] | None = None

    def load(self, name: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target, about, settings = resolve(name)
        self.model = self.grammar = None
        gc.collect()
        started = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(target)
        except (OSError, ValueError):  # weights saved without a tokenizer: the same seed's final has it
            self.tokenizer = AutoTokenizer.from_pretrained(str(Path(target).parent / "final"))
        dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=dtype).to(self.device).eval()
        self.settings, self.name, self.about = settings, name, about
        size = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[{name}] {about}\n[{size:.0f}M parameters on {self.device}, loaded in {time.time() - started:.0f} s; "
              f"max {self.limit()} new tokens, answer constraint {'on' if self.constrained() else 'off'}]")

    def limit(self) -> int:
        return self.max_new_tokens or self.settings.get("max_new_tokens", 512)

    def constrained(self) -> bool:
        return self.settings.get("constrain", False) if self.constrain is None else self.constrain

    def generate(self, prompt: str, puzzle: Puzzle | None) -> None:
        from transformers import (
            LogitsProcessor,
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
            TextStreamer,
        )

        if self.model is None:
            print("no model loaded: /m NAME (see /models)")
            return
        tokenizer, torch = self.tokenizer, self.torch
        self.last = (prompt, puzzle)
        encoded = tokenizer(prompt, return_tensors="pt").to(self.device)
        width = encoded["input_ids"].shape[1]
        processors, stops = LogitsProcessorList(), StoppingCriteriaList()
        if puzzle is not None:

            class StopAtClose(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return constrained.CLOSE in tokenizer.decode(input_ids[0, max(width, input_ids.shape[1] - 12):])

            stops.append(StopAtClose())
            if self.constrained():
                if self.grammar is None:
                    self.grammar = constrained.AnswerGrammar(constrained.token_strings(tokenizer), constrained.open_token_ids(tokenizer))
                request = constrained.RequestConstraint(self.grammar, puzzle.numbers, puzzle.ops, {})

                class Constrain(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        return request(input_ids[0, width:].tolist(), scores[0]).unsqueeze(0)

                processors.append(Constrain())
        # plain sampling at the given temperature, as the record samplers do (nothing from generation_config.json)
        sampling = {"do_sample": True, "temperature": self.temperature, "top_k": 0, "top_p": 1.0} if self.temperature > 0 else {"do_sample": False}
        started = time.time()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], repetition_penalty=1.0,
                max_new_tokens=self.limit(), logits_processor=processors, stopping_criteria=stops,
                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, **sampling,
            )
        new = out[0, width:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        took = time.time() - started
        status = f"{len(new)} tokens in {took:.0f} s ({len(new) / max(took, 1e-9):.1f}/s)"
        record = {"model": self.name, "about": self.about, "prompt": prompt, "completion": text, "tokens": len(new),
                  "seconds": round(took, 1), "temperature": self.temperature, "max_new_tokens": self.limit()}
        if puzzle is not None:
            cut = len(new) >= self.limit() and constrained.CLOSE not in text
            result = verify(text, puzzle)
            verdict = "CORRECT" if result.correct else ("MALFORMED" if result.malformed else "WRONG") + f" ({result.reason})"
            status += ("; cut off at the token limit" if cut else "") + f"; verifier: {verdict}"
            record |= {"puzzle": puzzle.to_dict() | {"ops": puzzle.ops}, "constrain": self.constrained(), "cut": cut,
                       "correct": result.correct, "malformed": result.malformed, "reason": result.reason}
            record["puzzle"].pop("solution", None)
        print(f"\n[{status}]", flush=True)
        if self.log:
            with open(self.log, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def puzzle(self, puzzle: Puzzle) -> None:
        level = f" ({puzzle.level})" if puzzle.level else ""
        print(f"[puzzle{level}: numbers {list(puzzle.numbers)}, target {puzzle.target}, operators {puzzle.ops}]")
        self.generate(format_prompt(puzzle, 0, "r1"), puzzle)

    def dev_puzzles(self) -> list[Puzzle]:
        if self.dev is None:
            self.dev = [Puzzle.from_dict(json.loads(line)) for line in DEV.read_text().splitlines() if line.strip()]
        return self.dev

    def random_puzzle(self, level: str) -> Puzzle:
        pool = [p for p in self.dev_puzzles() if not level or p.level == level.upper()]
        if not pool:
            raise ValueError(f"no dev puzzles at level {level!r} (L1 to L6)")
        return random.choice(pool)


def parse_puzzle(text: str) -> Puzzle:
    """`3 7 22 2 = 14` or `3 7 22 2 = 14 +-` (operators default to + - * /)."""
    left, _, right = text.partition("=")
    numbers = tuple(int(x) for x in left.replace(",", " ").split())
    rest = right.split()
    if len(numbers) < 2 or not rest:
        raise ValueError("write it as: /p 3 7 22 2 = 14   (optionally followed by the allowed operators, e.g. +-)")
    ops = rest[1] if len(rest) > 1 else ALL_OPS
    if not set(ops) <= set(ALL_OPS):
        raise ValueError(f"operators must be among {ALL_OPS}")
    return Puzzle(numbers, int(rest[0]), None, ops)


HELP = """commands:
  /models            list the models you can load
  /m NAME            load a model (a-base, a-best, a-collapsed, d-pre, d-best, a-005-s3, a path, ...)
  /p 3 7 22 2 = 14   give a puzzle (add operators after the target to restrict them, e.g. +-)
  /r [L1..L6]        a random puzzle from the dev ladder, optionally at one level
  /d N               dev-ladder puzzle number N (0-2399; 400 per level, L1 first)
  /again             run the last puzzle or text again (a new sample; also after /m, to compare models)
  /prompt            show the full prompt of the last run
  /t X               temperature (0 = greedy; records sample at 1)
  /n N               max new tokens (/n 0 = the model's own setting)
  /c on|off|auto     the answer constraint (auto = as the record was trained)
  /q                 quit
anything else is plain text for the model to continue (write \\n for a line break)"""


def handle(session: Session, line: str) -> bool:
    """Run one input line; False to quit."""
    command, _, arg = line.partition(" ")
    arg = arg.strip()
    if not line.startswith("/"):
        session.generate(line.replace("\\n", "\n"), None)
    elif command in ("/q", "/quit", "/exit"):
        return False
    elif command in ("/h", "/help", "/?"):
        print(HELP)
    elif command == "/models":
        print(list_models())
    elif command == "/m":
        session.load(arg)
    elif command == "/p":
        session.puzzle(parse_puzzle(arg))
    elif command == "/r":
        session.puzzle(session.random_puzzle(arg))
    elif command == "/d":
        session.puzzle(session.dev_puzzles()[int(arg)])
    elif command == "/again":
        if session.last is None:
            print("nothing to repeat yet")
        elif session.last[1] is not None:
            session.puzzle(session.last[1])
        else:
            session.generate(*session.last)
    elif command == "/prompt":
        print(session.last[0] if session.last else "nothing run yet")
    elif command == "/t":
        session.temperature = float(arg)
        print(f"[temperature {session.temperature}]")
    elif command == "/n":
        session.max_new_tokens = int(arg) or None
        print(f"[max {session.limit()} new tokens]")
    elif command == "/c":
        session.constrain = {"on": True, "off": False, "auto": None}[arg]
        print(f"[answer constraint {'on' if session.constrained() else 'off'}]")
    else:
        print(f"unknown command {command}; /help lists them")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="model to load at the start (see /models)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1, help="sampling seed (default: different every run)")
    parser.add_argument("--log", default="", help="append one JSON line per completion to this file")
    args = parser.parse_args()
    interactive = sys.stdin.isatty()
    if interactive:
        try:
            import readline  # noqa: F401  (line editing and history)
        except ImportError:
            pass
    session = Session(args.temperature, args.log)
    if args.seed >= 0:
        session.torch.manual_seed(args.seed)
        random.seed(args.seed)
    print(HELP if interactive else "", flush=True)
    if args.model:
        session.load(args.model)
    while True:
        try:
            line = input("\n> " if interactive else "").strip()
        except EOFError:
            break
        if not line:
            continue
        if not interactive:
            print(f"\n> {line}", flush=True)
        try:
            if not handle(session, line):
                break
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except (ValueError, KeyError, IndexError, OSError) as error:
            print(f"error: {error}")


if __name__ == "__main__":
    main()
