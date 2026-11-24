"""Probe base weights for written reasoning, without RL (U25). Timing-free; nothing is trained or kept but the outputs.

    python scripts/research/probe_reasoning.py --model <weights dir> --out-dir <dir> [--sets A,B,C] [--puzzles 200]
        [--samples 4] [--temperature 1.0] [--max-new 256] [--gsm8k <train.jsonl>] [--questions 100] [--batch 200]

Three prompt sets (LOG 11-21, the U25 plan):
  A  Countdown with track D's `r1` prompt (zero-shot), the first --puzzles dev puzzles, --samples samples each;
  B  the same puzzles with the `search` prompt (three worked trial-and-error examples in front);
  C  GSM8K word problems (--questions from the train file, 4 other worked questions in front), greedy.
Prompts are tokenized as GRPO does (the tokenizer's default special tokens, so BOS first). Sampling is plain:
temperature --temperature, no top-k or top-p. A completion is cut after its first `</answer>` (A, B) or before the
next `Question:` (C), and before a new `User:` turn. Per set it writes:
  - step accuracy: the share of written `a op b = c` steps (any run of numbers and + - * / with brackets, then
    `= number`) whose arithmetic is right;
  - grounded-chain rate: the share of samples with at least two right steps in which every operand is a given number
    (the puzzle's, or the question's for C) or the result of an earlier such step;
  - solve rate (A, B: `verify` as in training; C: the number after "The answer is", else the last number);
  - well-formed rate (A, B: an `<answer>` block that parses; C: "The answer is N");
  - median completion length in tokens, and the loop rate (a word 4-gram repeated four or more times).
Outputs in --out-dir: probe.json (numbers), samples.md (the first 20 puzzles' or questions' first samples, for
reading by eye) and completions.jsonl (every completion with its scores).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from rlvr_speedrun.countdown import ANSWER_RE, Puzzle, format_prompt, verify

NUM = r"-?\d+(?:\.\d+)?"
_RHS = re.compile(r"=\s*(" + NUM + r")")
_LHS_CHARS = set("0123456789. +-*/()")
_GIVEN = re.compile(r"\d+(?:\.\d+)?")


def _normalise(text: str) -> str:
    """Operators and numbers in one spelling: x and the typographic signs as * and /, no $ or thousands commas, and
    full stops that end a sentence turned into a bar so a step never runs across sentences."""
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("$", "")
    text = re.sub(r"(?<=\d)\s+[xX]\s+(?=\d)", " * ", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\.(?!\d)", " | ", text)


def _eval(node: ast.AST, leaves: list[Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body, leaves)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = Fraction(str(node.value))
        leaves.append(value)
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, leaves)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left, leaves), _eval(node.right, leaves)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise ZeroDivisionError
        return a / b
    raise ValueError("not arithmetic")


def _close(value: Fraction, rhs: str) -> bool:
    """Right if exact, or within one unit of the last decimal place the writer gave."""
    target = Fraction(rhs)
    if value == target:
        return True
    places = len(rhs.split(".")[1]) if "." in rhs else 0
    return places > 0 and abs(value - target) <= Fraction(1, 10**places)


def steps(text: str) -> list[dict]:
    """Every written step `expression = number` in order: its operands, the written result and whether it is right."""
    text = _normalise(text)
    out = []
    for m in _RHS.finditer(text):
        i = m.start()
        while i > 0 and text[i - 1] in _LHS_CHARS:
            i -= 1
        lhs = text[i : m.start()].strip()
        while lhs.count("(") > lhs.count(")") and lhs.startswith("("):
            lhs = lhs[1:].strip()
        while lhs.count(")") > lhs.count("(") and lhs.endswith(")"):
            lhs = lhs[:-1].strip()
        if not lhs or not re.search(r"\d\s*\)?\s*[-+*/]\s*\(?\s*-?\d", lhs):
            continue
        leaves: list[Fraction] = []
        try:
            value = _eval(ast.parse(lhs, mode="eval"), leaves)
        except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
            continue
        if len(leaves) < 2:
            continue
        out.append({"lhs": lhs, "rhs": m.group(1), "leaves": leaves, "right": _close(value, m.group(1))})
    return out


def grounded_chain(step_list: list[dict], given) -> int:
    """How many right steps use only given numbers and results of earlier such steps (in order)."""
    available = {Fraction(str(g)) for g in given}
    n = 0
    for step in step_list:
        if step["right"] and all(leaf in available for leaf in step["leaves"]):
            n += 1
            available.add(Fraction(step["rhs"]))
    return n


def loops(text: str, n: int = 4, times: int = 4) -> bool:
    words = text.split()
    return any(c >= times for c in Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1)).values())


def cut_countdown(text: str) -> str:
    end = text.find("</answer>")
    if end >= 0:
        text = text[: end + len("</answer>")]
    turn = text.find("\nUser:")
    return text[:turn] if turn >= 0 else text


def cut_gsm8k(text: str) -> str:
    for stop in ("\nQuestion:", "Question:", "\nUser:"):
        at = text.find(stop)
        if at >= 0:
            text = text[:at]
    return text.rstrip()


_ANSWER_IS = re.compile(r"The answer is\s*\$?\s*(" + NUM + r")", re.IGNORECASE)


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_shot(question: str, answer: str) -> str:
    """A worked GSM8K answer as plain prose: calculator marks removed, `#### N` written as "The answer is N."."""
    work, _, final = answer.partition("####")
    work = re.sub(r"<<[^>]*>>", "", work).strip()
    return f"Question: {question}\nAnswer: {work}\nThe answer is {final.strip()}.\n\n"


def score_gsm8k(text: str, gold: str) -> dict:
    plain = _normalise(text).replace(" | ", " ")
    m = _ANSWER_IS.search(plain)
    numbers = re.findall(NUM, plain)
    final = m.group(1) if m else (numbers[-1] if numbers else None)
    try:
        solved = final is not None and Fraction(final) == Fraction(gold)
    except (ValueError, ZeroDivisionError):
        solved = False
    return {"well_formed": m is not None, "solved": bool(solved), "final": final}


def score_countdown(text: str, puzzle: Puzzle) -> dict:
    result = verify(text, puzzle)
    return {"well_formed": bool(ANSWER_RE.search(text)) and not result.malformed, "solved": result.correct, "reason": result.reason}


def summarise(rows: list[dict]) -> dict:
    n_steps = sum(r["n_steps"] for r in rows)
    return {
        "samples": len(rows),
        "steps_per_sample": n_steps / len(rows),
        "step_accuracy": sum(r["n_right"] for r in rows) / n_steps if n_steps else None,
        "grounded_chain_rate": sum(r["chain"] >= 2 for r in rows) / len(rows),
        "solve_rate": sum(r["solved"] for r in rows) / len(rows),
        "well_formed_rate": sum(r["well_formed"] for r in rows) / len(rows),
        "median_tokens": statistics.median(r["tokens"] for r in rows),
        "loop_rate": sum(r["loop"] for r in rows) / len(rows),
    }


def _generate(model, tok, prompts: list[str], samples: int, temperature: float, max_new: int, batch: int) -> list[list[str]]:
    import torch

    expanded = [p for p in prompts for _ in range(samples)]
    texts: list[str] = []
    for i in range(0, len(expanded), batch):
        enc = tok(expanded[i : i + batch], return_tensors="pt", padding=True).to("cuda")
        kw = {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [texts[j * samples : (j + 1) * samples] for j in range(len(prompts))]


def _row(text: str, given, score: dict, tok) -> dict:
    found = steps(text)
    return {"text": text, "n_steps": len(found), "n_right": sum(s["right"] for s in found), "chain": grounded_chain(found, given),
            "tokens": len(tok.encode(text, add_special_tokens=False)), "loop": loops(text), **score}


def _markdown(name: str, items: list[tuple[str, dict]]) -> str:
    lines = [f"## Set {name}\n"]
    for k, (header, row) in enumerate(items):
        flags = (f"steps {row['n_steps']}, right {row['n_right']}, chain {row['chain']}, solved {row['solved']}, "
                 f"well-formed {row['well_formed']}, loop {row['loop']}, {row['tokens']} tokens")
        lines.append(f"### {name}{k + 1}: {header}\n\n{flags}\n\n````\n{row['text']}\n````\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sets", default="A,B,C")
    parser.add_argument("--dev", default="data/ladder_v3_dev.jsonl")
    parser.add_argument("--puzzles", type=int, default=200)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--gsm8k", default=str(Path.home() / "orcd/pool/data/gsm8k/train.jsonl"))  # also on a box
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--context", type=int, default=1024, help="prompt + new tokens must fit (the pretraining length)")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from rlvr_speedrun.model_utils import load_tokenizer

    torch.manual_seed(args.seed)
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "gpu": torch.cuda.get_device_name(), "settings": {k: v for k, v in vars(args).items() if k != "model"}}
    md = [f"# Reasoning probe: {args.model}\n"]
    every: list[dict] = []
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    with open(args.dev) as f:
        puzzles = [Puzzle.from_dict(json.loads(line)) for line in f][: args.puzzles]
    for name in sets:
        t0 = time.perf_counter()
        if name in ("A", "B"):
            style = "r1" if name == "A" else "search"
            prompts = [format_prompt(p, style=style) for p in puzzles]
            done = _generate(model, tok, prompts, args.samples, args.temperature, args.max_new, args.batch)
            rows, shown = [], []
            for j, (p, texts) in enumerate(zip(puzzles, done)):
                for s, raw in enumerate(texts):
                    text = cut_countdown(raw)
                    row = _row(text, p.numbers, score_countdown(text, p), tok)
                    rows.append({"set": name, "item": j, "sample": s, **row})
                    if s == 0 and j < args.show:
                        shown.append((f"numbers {list(p.numbers)}, target {p.target}, ops {p.ops}", row))
        elif name == "C":
            with open(args.gsm8k) as f:
                data = [json.loads(line) for line in f]
            shots = "".join(gsm8k_shot(d["question"], d["answer"]) for d in data[:4])
            prompts, golds, questions, skipped = [], [], [], 0
            for d in data[4:]:
                prompt = shots + f"Question: {d['question']}\nAnswer:"
                if len(tok.encode(prompt)) + args.max_new > args.context:
                    skipped += 1
                    continue
                prompts.append(prompt)
                golds.append(gsm8k_gold(d["answer"]))
                questions.append(d["question"])
                if len(prompts) == args.questions:
                    break
            summary["C_skipped_for_length"] = skipped
            done = _generate(model, tok, prompts, 1, 0.0, args.max_new, args.batch)
            rows, shown = [], []
            for j, (q, gold, texts) in enumerate(zip(questions, golds, done)):
                text = cut_gsm8k(texts[0])
                given = [Fraction(x) for x in _GIVEN.findall(_normalise(q).replace(" | ", " "))]
                row = _row(text, given, score_gsm8k(text, gold), tok)
                rows.append({"set": name, "item": j, "sample": 0, "gold": gold, **row})
                if j < args.show:
                    shown.append((f"{q} (answer {gold})", row))
        else:
            raise SystemExit(f"unknown set {name!r}; choose from A, B, C")
        summary[name] = {**summarise(rows), "seconds": time.perf_counter() - t0}
        print(name, json.dumps(summary[name]), flush=True)
        md.append(_markdown(name, shown))
        every += rows
    (out / "probe.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "samples.md").write_text("\n".join(md))
    with open(out / "completions.jsonl", "w") as f:
        for row in every:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
