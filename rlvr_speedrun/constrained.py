"""Answer-constrained decoding (`--constrain-answer` in training, `--constrain` in `ladder_confirm`).

Free text is unconstrained until the completion contains `<answer>`. From there on, only tokens
that keep the text a prefix of `EXPR</answer>` may be sampled, where EXPR uses each of the
puzzle's numbers exactly once, only the puzzle's operators (binary, no unary minus) and
balanced parentheses (no two spaces in a row, and no more open parentheses than operators still
to come, so answers stay short). So every answer is a well-formed use of the right numbers, and the
verifier's reward only says whether it hits the target.

The same grammar runs on both sides. vLLM masks the logits while sampling
(`AnswerLogitsProcessor`), and the trainer renormalises its log-probs over the same allowed
tokens (`allowed_per_position`), so the policy gradient is the one of the constrained policy.

With a think budget, a completion that has not opened the answer after that many tokens is forced
to write OPEN next (one allowed token per position, so no gradient flows through it). Thinking then
never ends in a truncated, malformed answer: without it, any free text risks the 256-token cap and
the malformed penalty, which teaches the model to answer at once whether or not thinking helps.
"""

from __future__ import annotations

OPEN, CLOSE = "<answer>", "</answer>"
_CHARS = set("0123456789+-*/() ") | set(CLOSE)

# state: (remaining numbers as a sorted tuple of strings, depth, mode, partial number, matched chars of CLOSE, last char was a space)
# mode: "operand" (expect a number or "("), "number" (inside a number), "operator", "close", "done"
State = tuple


def initial_state(numbers) -> State:
    return (tuple(sorted(str(n) for n in numbers)), 0, "operand", "", 0, False)


def step(state: State, ch: str, ops: str) -> State | None:
    """The state after one more character, or None if no valid answer continues this way."""
    remaining, depth, mode, partial, closed, space = state
    if mode == "number":
        if ch.isdigit():
            grown = partial + ch
            return (remaining, depth, mode, grown, 0, False) if any(r.startswith(grown) for r in remaining) else None
        if partial not in remaining:
            return None
        rest = list(remaining)
        rest.remove(partial)
        return step((tuple(rest), depth, "operator", "", 0, False), ch, ops)
    if ch == " " and mode in ("operand", "operator"):
        return None if space else (remaining, depth, mode, "", 0, True)
    if mode == "operand":
        if ch == "(" and depth + 1 < len(remaining):
            return (remaining, depth + 1, mode, "", 0, False)
        if ch.isdigit() and any(r.startswith(ch) for r in remaining):
            return (remaining, depth, "number", ch, 0, False)
        return None
    if mode == "operator":
        if ch == ")" and depth > 0:
            return (remaining, depth - 1, mode, "", 0, False)
        if ch in ops and remaining:
            return (remaining, depth, "operand", "", 0, False)
        if ch == CLOSE[0] and depth == 0 and not remaining:
            return (remaining, depth, "close", "", 1, False)
        return None
    if mode == "close" and ch == CLOSE[closed]:
        return (remaining, depth, "done" if closed + 1 == len(CLOSE) else "close", "", closed + 1, False)
    return None


def feed(state: State | None, text: str, ops: str) -> State | None:
    for ch in text:
        if state is None:
            return None
        state = step(state, ch, ops)
    return state


def token_strings(tokenizer) -> list[str]:
    """The text each token id adds to a completion (leading space included)."""
    out = []
    special = set(tokenizer.all_special_ids)
    for i, tok in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if tok is None or i in special:
            out.append("")
            continue
        text = tokenizer.convert_tokens_to_string([tok])
        if tok.startswith("▁") and not text.startswith(" "):  # sentencepiece drops the leading space of a lone token
            text = " " + text
        out.append(text)
    return out


def open_token_ids(tokenizer) -> tuple[int, ...]:
    """The ids that spell OPEN after free text (a newline first, so it never merges with the text before)."""
    ids = tokenizer.encode("\n" + OPEN, add_special_tokens=False)
    return tuple(ids)


class AnswerGrammar:
    """Token-level view of the grammar for one tokenizer, with allowed sets cached per state."""

    def __init__(self, strings: list[str], open_ids: tuple[int, ...] = ()):
        self.strings = strings
        self.open_ids = tuple(open_ids)  # the tokens of OPEN, forced once a think budget runs out
        self.candidates = [i for i, s in enumerate(strings) if s and set(s) <= _CHARS]
        self._cache: dict = {}

    def text(self, token_id: int) -> str:
        """The token's text; "" for ids past the tokenizer (models pad their embedding matrix, e.g. Qwen3)."""
        return self.strings[token_id] if token_id < len(self.strings) else ""

    def allowed(self, state: State, ops: str) -> tuple[int, ...]:
        key = (state, ops)
        if key not in self._cache:
            self._cache[key] = tuple(i for i in self.candidates if feed(state, self.strings[i], ops) is not None)
        return self._cache[key]

    def forced(self, position: int, think_budget: int) -> tuple[int, ...] | None:
        """Before the answer opens: the single allowed id at `position` once the think budget is spent, else None."""
        k = position - think_budget
        return (self.open_ids[k],) if think_budget and 0 <= k < len(self.open_ids) else None

    def allowed_per_position(self, token_ids: list[int], numbers, ops: str, think_budget: int = 0) -> list[tuple[int, ...] | None]:
        """For each completion position, the allowed ids when that token was sampled (None = unconstrained)."""
        out, text, state = [], "", None
        for position, tid in enumerate(token_ids):
            if state is None:
                out.append(self.forced(position, think_budget))
                text += self.text(tid)
                if OPEN in text:
                    state = feed(initial_state(numbers), text.split(OPEN, 1)[1], ops) or False
            elif state is False or state[2] == "done":  # opening token carried invalid text / answer complete
                out.append(None)
            else:
                allowed = self.allowed(state, ops)
                out.append(allowed or None)
                state = feed(state, self.text(tid), ops) or False
        return out


class RequestConstraint:
    """Per-request vLLM logits processor: `(output_ids, logits) -> logits`, incremental over output_ids."""

    def __init__(self, grammar: AnswerGrammar, numbers, ops: str, index_cache: dict, think_budget: int = 0):
        self.grammar, self.numbers, self.ops, self.index_cache = grammar, tuple(numbers), ops, index_cache
        self.think_budget = think_budget
        self.seen, self.text, self.state = 0, "", None

    def __call__(self, output_ids: list[int], logits):
        import torch

        for tid in output_ids[self.seen :]:
            if self.state is None:
                self.text += self.grammar.text(tid)
                if OPEN in self.text:
                    self.state = feed(initial_state(self.numbers), self.text.split(OPEN, 1)[1], self.ops) or False
            elif self.state is not False:
                self.state = feed(self.state, self.grammar.text(tid), self.ops) or False
        self.seen = len(output_ids)
        if self.state is None:
            allowed = self.grammar.forced(len(output_ids), self.think_budget)
        elif not self.state or self.state[2] == "done":
            return logits
        else:
            allowed = self.grammar.allowed(self.state, self.ops)
        if not allowed:
            return logits
        index = self.index_cache.get(allowed)
        if index is None:
            index = self.index_cache[allowed] = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[index] = logits[index]
        return masked


def mask_logits(logits, rows: list[tuple[int, tuple[int, ...]]]):
    """Set every logit outside the allowed ids to -inf for the given (row, allowed ids) pairs."""
    import torch

    if not rows:
        return logits
    r = torch.tensor([row for row, ids in rows for _ in ids], dtype=torch.long, device=logits.device)
    c = torch.tensor([i for _, ids in rows for i in ids], dtype=torch.long, device=logits.device)
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[r, c] = True
    constrained = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    constrained[torch.tensor([row for row, _ in rows], device=logits.device)] = True
    return logits.masked_fill(constrained.unsqueeze(1) & ~keep, float("-inf"))

