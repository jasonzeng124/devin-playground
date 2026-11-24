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


_DIGITS = frozenset("0123456789")


class AnswerGrammar:
    """Token-level view of the grammar for one tokenizer, with allowed sets cached per state.

    `factored` builds a state's allowed set from parts instead of walking every candidate (`_allowed_factored`): the
    same sets, at a fraction of the cost for a state not seen before. Training puzzles are new at every step and a
    state holds the puzzle's remaining numbers, so most of a native sampling step's grammar time is such states.
    """

    def __init__(self, strings: list[str], open_ids: tuple[int, ...] = (), factored: bool = False, interned: bool = False):
        self.strings = strings
        self.open_ids = tuple(open_ids)  # the tokens of OPEN, forced once a think budget runs out
        self._forced = tuple((i,) for i in self.open_ids)  # one tuple each, so `MaskTable` finds them by identity
        self.candidates = [i for i, s in enumerate(strings) if s and set(s) <= _CHARS]
        self._cache: dict = {}
        self._trie: dict = {}  # the candidates' strings as a character trie; None keys hold the ids ending there
        for i in self.candidates:
            node = self._trie
            for ch in strings[i]:
                node = node.setdefault(ch, {})
            node.setdefault(None, []).append(i)
        self.factored = factored
        if factored:  # the candidates by kind: no digit, only digits (by string), both
            self._plain: dict = {}
            self._plain_trie: dict = {}
            self._mixed_trie: dict = {}
            self._digit_ids: dict[str, list[int]] = {}
            for i in self.candidates:
                digits = sum(ch in _DIGITS for ch in strings[i])
                if digits == len(strings[i]):
                    self._digit_ids.setdefault(strings[i], []).append(i)
                    continue
                node = self._mixed_trie if digits else self._plain_trie
                for ch in strings[i]:
                    node = node.setdefault(ch, {})
                node.setdefault(None, []).append(i)
            self._digit_lengths = sorted({len(s) for s in self._digit_ids})
        # `interned`: `RequestConstraint` keeps an int per (state, ops) and steps by (int, token id) lookups, so a
        # token another row already took from the same state costs one dict lookup instead of `feed` + `allowed`
        self.interned = interned
        self._ids: dict = {}  # (state, ops) -> its int
        self._info: list = []  # int -> (state, ops, allowed ids; None once done)
        self._next: dict = {}  # (int, token id) -> the next int (-1: the token breaks the answer)

    def intern(self, state: State, ops: str) -> int:
        key = (state, ops)
        sid = self._ids.get(key)
        if sid is None:
            sid = self._ids[key] = len(self._info)
            self._info.append((state, ops, None if state[2] == "done" else self.allowed(state, ops)))
        return sid

    def advance(self, sid: int, token_id: int) -> int:
        """The interned state after `token_id` from interned state `sid`; -1 if the grammar rejects it."""
        key = (sid, token_id)
        nxt = self._next.get(key)
        if nxt is None:
            state, ops, _ = self._info[sid]
            after = feed(state, self.text(token_id), ops)
            nxt = self._next[key] = self.intern(after, ops) if after else -1
        return nxt

    def trim(self, limit: int) -> None:
        """Drop every cached state once more than `limit` are held (training puzzles are new at every step, so the
        caches otherwise grow without bound). Pure caches: later results are the same. Call it only while no
        `RequestConstraint` is mid-request, since their ints are dropped too."""
        if len(self._info) + len(self._cache) > limit:
            self._ids, self._info, self._next, self._cache = {}, [], {}, {}

    def text(self, token_id: int) -> str:
        """The token's text; "" for ids past the tokenizer (models pad their embedding matrix, e.g. Qwen3)."""
        return self.strings[token_id] if token_id < len(self.strings) else ""

    def allowed(self, state: State, ops: str) -> tuple[int, ...]:
        key = (state, ops)
        if key not in self._cache:
            if self.factored and _DIGITS.isdisjoint(ops):
                self._cache[key] = self._allowed_factored(state, ops)
            else:
                # the candidates that `feed` accepts from `state`, ascending, found by walking the trie so that a
                # prefix the grammar rejects drops every token under it at once (PLAN D7: ~11x faster than testing
                # each one)
                self._cache[key] = tuple(sorted(self._walk(self._trie, state, ops)))
        return self._cache[key]

    def _walk(self, trie: dict, state: State, ops: str) -> list[int]:
        """The ids in `trie` whose strings `feed` accepts from `state`."""
        out, stack = [], [(trie, state)]
        while stack:
            node, at = stack.pop()
            for ch, child in node.items():
                if ch is not None and (after := step(at, ch, ops)) is not None:
                    out.extend(child.get(None, ()))
                    stack.append((child, after))
        return out

    def _allowed_factored(self, state: State, ops: str) -> tuple[int, ...]:
        """What the walk over every candidate gives, from three kinds of candidates that `step` treats differently
        (`ops` must hold no digit):

        - No digit in the token. Along such a string `step` never reads the numbers themselves, only the mode, how
          many numbers remain, the depth, the place in CLOSE and the space flag. Inside a number it first asks
          whether the number is complete, and then carries on as the operator state with that number used. So
          these ids are cached per that shape and walked once for each.
        - Only digits. `step` takes them where a number may start or go on, and then exactly when the digits so
          far plus the token are a prefix of a remaining number, so they are read off the remaining numbers.
        - Both (none in some vocabularies). Walked from the state as before.
        """
        remaining, depth, mode, partial, closed, space = state
        if mode == "number":
            shape = ("operator", len(remaining) - 1, depth, 0, False) if partial in remaining else None
        else:
            shape = (mode, len(remaining), depth, closed, space)
        plain: tuple[int, ...] = ()
        if shape is not None:
            found = self._plain.get((shape, ops))
            if found is None:
                found = self._plain[(shape, ops)] = tuple(sorted(self._walk(self._plain_trie, state, ops)))
            plain = found
        out: list[int] = []
        if mode == "operand" or mode == "number":
            head = partial if mode == "number" else ""
            pieces = set()
            for number in remaining:
                if number.startswith(head):
                    tail = number[len(head) :]
                    for k in self._digit_lengths:
                        if k > len(tail):
                            break
                        pieces.add(tail[:k])
            for piece in pieces:
                out.extend(self._digit_ids.get(piece, ()))
        if self._mixed_trie:
            out.extend(self._walk(self._mixed_trie, state, ops))
        if not out:
            return plain
        out.extend(plain)
        return tuple(sorted(out))

    def forced(self, position: int, think_budget: int) -> tuple[int, ...] | None:
        """Before the answer opens: the single allowed id at `position` once the think budget is spent, else None."""
        k = position - think_budget
        return self._forced[k] if think_budget and 0 <= k < len(self.open_ids) else None

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
        self.sid = None  # with an interned grammar: None before OPEN, -1 broken, else the interned state

    def __call__(self, output_ids: list[int], logits):
        import torch

        allowed = self.allowed_ids(output_ids)
        if not allowed:
            return logits
        index = self.index_cache.get(allowed)
        if index is None:
            index = self.index_cache[allowed] = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[index] = logits[index]
        return masked

    def allowed_ids(self, output_ids: list[int]) -> tuple[int, ...] | None:
        """Advance over the newly sampled ids; the ids allowed next (None or () = unconstrained)."""
        if self.grammar.interned:
            return self._allowed_interned(output_ids)
        for tid in output_ids[self.seen :]:
            if self.state is None:
                self.text += self.grammar.text(tid)
                if OPEN in self.text:
                    self.state = feed(initial_state(self.numbers), self.text.split(OPEN, 1)[1], self.ops) or False
            elif self.state is not False:
                self.state = feed(self.state, self.grammar.text(tid), self.ops) or False
        self.seen = len(output_ids)
        if self.state is None:
            return self.grammar.forced(len(output_ids), self.think_budget)
        if not self.state or self.state[2] == "done":
            return None
        return self.grammar.allowed(self.state, self.ops)

    def _allowed_interned(self, output_ids: list[int]) -> tuple[int, ...] | None:
        """`allowed_ids` over the grammar's interned states: the same ids."""
        grammar = self.grammar
        for tid in output_ids[self.seen :]:
            if self.sid is None:
                self.text += grammar.text(tid)
                if OPEN in self.text:
                    state = feed(initial_state(self.numbers), self.text.split(OPEN, 1)[1], self.ops)
                    self.sid = grammar.intern(state, self.ops) if state else -1
            elif self.sid >= 0:
                self.sid = grammar.advance(self.sid, tid)
        self.seen = len(output_ids)
        if self.sid is None:
            return grammar.forced(len(output_ids), self.think_budget)
        return grammar._info[self.sid][2] if self.sid >= 0 else None


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



def mask_rows_cached(logits, rows: list[tuple[int, tuple[int, ...]]], index_cache: dict):
    """`mask_logits` for a whole batch in a fixed handful of kernels: each allowed set's ids live on the device
    in `index_cache` (keyed by the tuple), so a call costs one concatenation, not a mask per row."""
    import torch

    rows = [(row, ids) for row, ids in rows if ids]
    if not rows:
        return logits
    parts = []
    for _, ids in rows:
        index = index_cache.get(ids)
        if index is None:
            index = index_cache[ids] = torch.tensor(ids, dtype=torch.long, device=logits.device)
        parts.append(index)
    c = torch.cat(parts)
    row_ids = torch.tensor([row for row, _ in rows], dtype=torch.long).to(logits.device, non_blocking=True)
    lengths = torch.tensor([len(ids) for _, ids in rows], dtype=torch.long).to(logits.device, non_blocking=True)
    r = torch.repeat_interleave(row_ids, lengths, output_size=c.numel())
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[r, c] = True
    constrained = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    constrained[row_ids] = True
    return logits.masked_fill(constrained.unsqueeze(1) & ~keep, float("-inf"))


class MaskTable:
    """`mask_rows_cached` with each allowed set as a row of a device table of blocked ids: a call gathers one table
    row per logits row (row 0 blocks nothing) and fills them with -inf, so it costs one small copy of row slots, one
    copy for sets not seen before, and two kernels, whatever the rows' sets. Sets are found by identity (the grammar
    returns its cached tuple for a state); when the table is full it starts over. Gives the same logits."""

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.blocked = None  # [capacity, vocab] bool, made on the first call
        self.slots: dict = {}  # id(ids) -> (ids, slot); holding `ids` keeps its id from being reused

    def __call__(self, logits, rows: list[tuple[int, tuple[int, ...]]]):
        import torch

        rows = [(row, ids) for row, ids in rows if ids]
        if not rows:
            return logits
        if self.blocked is None or self.blocked.shape[1] != logits.shape[1] or self.blocked.device != logits.device:
            self.blocked = torch.zeros(self.capacity, logits.shape[1], dtype=torch.bool, device=logits.device)
            self.slots = {}
        new: dict = {}
        for _, ids in rows:
            hit = self.slots.get(id(ids))
            if (hit is None or hit[0] is not ids) and id(ids) not in new:
                new[id(ids)] = ids
        if len(self.slots) + len(new) >= self.capacity:  # start over: every set of this call gets a fresh slot
            self.slots = {}
            new = {id(ids): ids for _, ids in rows}
            if len(new) >= self.capacity:
                raise ValueError(f"{len(new)} distinct allowed sets in one call; MaskTable capacity is {self.capacity}")
        if new:
            first = len(self.slots) + 1
            slots, write_rows, write_cols = [], [], []
            for k, (key, ids) in enumerate(new.items()):
                self.slots[key] = (ids, first + k)
                slots.append(first + k)
                write_rows.extend([first + k] * len(ids))
                write_cols.extend(ids)
            dev = logits.device
            fresh = torch.tensor(slots, dtype=torch.long).to(dev, non_blocking=True)
            self.blocked[fresh] = True
            self.blocked[torch.tensor(write_rows, dtype=torch.long).to(dev, non_blocking=True),
                         torch.tensor(write_cols, dtype=torch.long).to(dev, non_blocking=True)] = False
        slot_of = [0] * logits.shape[0]
        for row, ids in rows:
            slot_of[row] = self.slots[id(ids)][1]
        index = torch.tensor(slot_of, dtype=torch.long).to(logits.device, non_blocking=True)
        return logits.masked_fill(self.blocked[index], float("-inf"))
