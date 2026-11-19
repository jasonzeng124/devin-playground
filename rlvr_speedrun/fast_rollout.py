"""Same-process sampler for Llama-class policies (`--rollout-backend native`; INBOX 11-17 14:59, LOG 11-18 01:10).

r96's profile showed that for track D's 110M model vLLM's per-request bookkeeping (scheduler, output processing,
detokenizer), not the forward pass, sets the sampling time. This sampler keeps a bf16 copy of the policy's weights
in the training process and decodes the whole batch in lock step:

- each distinct prompt is prefilled once and its keys and values are copied to its `n` samples;
- a decode step (embedding, the layers, the head) is captured in a CUDA graph per (batch bucket, key-length
  bucket) and replayed; the keys and values live in a static cache;
- the answer constraint is the same `RequestConstraint` grammar the vLLM processor uses, applied with
  `mask_rows_cached`; the stop rules are vLLM's here (`</answer>` in the completion text, or an EOS id, both kept
  in the output; or `max_tokens`);
- sampling is exact for the temperature-scaled, masked distribution (probabilities over exponential noise, as vLLM
  samples); temperature 0 is argmax.

`generate` returns what `VLLMRollout.generate` returns (prompt ids as encoded, then right-padded completions, and
each completion's length), so the loss code is unchanged; as with vLLM, the loss recomputes log-probs with the
training model. `sync(model)` copies the training weights in after every optimizer step.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .constrained import CLOSE, AnswerGrammar, RequestConstraint, mask_rows_cached, open_token_ids, token_strings


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


class NativeRollout:
    def __init__(self, model, tokenizer, max_len: int, seed: int, dtype: torch.dtype = torch.bfloat16, constrain_answer: bool = False,
                 cuda_graphs: bool = True, batch_bucket: int = 64, key_bucket: int = 32, factored_grammar: bool = False):
        model = _unwrap(model)
        config = model.config
        if config.model_type != "llama":
            raise ValueError(f"rollout_backend='native' supports Llama models only, got {config.model_type!r}")
        rope = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
        if getattr(config, "attention_bias", False) or getattr(config, "mlp_bias", False) or rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("rollout_backend='native' supports Llama without biases or RoPE scaling")
        rope_theta = rope.get("rope_theta") or getattr(config, "rope_theta", None) or 10000.0
        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError(f"rollout_backend='native' needs hidden_act='silu', got {config.hidden_act!r}")
        self.device = next(model.parameters()).device
        self.dtype = dtype
        self.layers_n = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads or self.heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.heads
        self.eps = config.rms_norm_eps
        self.max_len = max_len
        self.batch_bucket, self.key_bucket = batch_bucket, key_bucket
        self.cuda_graphs = cuda_graphs and self.device.type == "cuda"
        eos = {tokenizer.eos_token_id}
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        self.eos = frozenset(e for e in eos if e is not None)
        self.strings = token_strings(tokenizer)
        self.grammar = AnswerGrammar(self.strings, open_token_ids(tokenizer), factored=factored_grammar) if constrain_answer else None
        self.index_cache: dict = {}
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.mask_cpu = 0.0  # the constraint's CPU seconds in the last `generate` call
        self.last_allowed = None  # per row of the last constrained `generate`: the allowed ids at each position
        self.stats: dict = {}

        # bf16 copies of the weights; q/k/v and gate/up are fused so a layer is four matmuls
        def buf(*shape):
            return torch.empty(*shape, dtype=dtype, device=self.device)

        hidden, inter = config.hidden_size, config.intermediate_size
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.embed = buf(config.vocab_size, hidden)
        self.tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
        self.head = self.embed if self.tied else buf(config.vocab_size, hidden)
        self.final_norm = buf(hidden)
        self.layers = [dict(ln1=buf(hidden), qkv=buf(q + 2 * kv, hidden), o=buf(hidden, q), ln2=buf(hidden),
                            gate_up=buf(2 * inter, hidden), down=buf(hidden, inter)) for _ in range(self.layers_n)]
        self.inter = inter
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32, device=self.device), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos, self.sin = emb.cos().to(dtype), emb.sin().to(dtype)
        self.sync(model)

        self.capacity = 0  # rows the static cache holds
        self.graphs: dict = {}
        self.pool = None

    @torch.no_grad()
    def sync(self, model) -> None:
        model = _unwrap(model)
        inner = model.model
        self.embed.copy_(inner.embed_tokens.weight)
        if not self.tied:
            self.head.copy_(model.lm_head.weight)
        self.final_norm.copy_(inner.norm.weight)
        for mine, layer in zip(self.layers, inner.layers):
            attn, mlp = layer.self_attn, layer.mlp
            mine["ln1"].copy_(layer.input_layernorm.weight)
            mine["ln2"].copy_(layer.post_attention_layernorm.weight)
            q, k = attn.q_proj.weight.shape[0], attn.k_proj.weight.shape[0]
            mine["qkv"][:q].copy_(attn.q_proj.weight)
            mine["qkv"][q : q + k].copy_(attn.k_proj.weight)
            mine["qkv"][q + k :].copy_(attn.v_proj.weight)
            mine["o"].copy_(attn.o_proj.weight)
            mine["gate_up"][: self.inter].copy_(mlp.gate_proj.weight)
            mine["gate_up"][self.inter :].copy_(mlp.up_proj.weight)
            mine["down"].copy_(mlp.down_proj.weight)

    # ---- the model -------------------------------------------------------------------------------------------

    def _norm(self, x, weight):
        xf = x.float()
        return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

    def _rope(self, x, cos, sin):  # x [B, H, T, D]; cos/sin [B, 1, T, D]
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin

    def _layers(self, h, positions, write_kv, attend):
        """h [B, T, hidden]; positions [B, T]; write_kv(layer, k, v) stores keys/values, attend(layer, q) -> [B, H, T, D]."""
        b, t, _ = h.shape
        cos, sin = self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)
        nq, nkv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        for i, w in enumerate(self.layers):
            x = self._norm(h, w["ln1"])
            qkv = x @ w["qkv"].T
            q = qkv[..., :nq].view(b, t, self.heads, self.head_dim).transpose(1, 2)
            k = qkv[..., nq : nq + nkv].view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
            v = qkv[..., nq + nkv :].view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            write_kv(i, k, v)
            out = attend(i, q).transpose(1, 2).reshape(b, t, nq)
            h = h + out @ w["o"].T
            x = self._norm(h, w["ln2"])
            gate, up = (x @ w["gate_up"].T).split(self.inter, dim=-1)
            h = h + (F.silu(gate) * up) @ w["down"].T
        return self._norm(h, self.final_norm)

    def _sdpa(self, q, k, v, mask):
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _ensure_capacity(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        self.graphs.clear()
        self.pool = None
        self.capacity = rows
        shape = (self.layers_n, rows, self.kv_heads, self.max_len, self.head_dim)
        self.k_cache = self.v_cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.valid = torch.zeros(rows, self.max_len, dtype=torch.bool, device=self.device)
        self.tokens = torch.zeros(rows, dtype=torch.long, device=self.device)
        self.positions = torch.zeros(rows, dtype=torch.long, device=self.device)
        self.slot = torch.zeros(1, dtype=torch.long, device=self.device)

    def _decode(self, rows: int, keys: int):
        """One decode step for the first `rows` rows: feeds self.tokens at self.positions into cache slot self.slot,
        attends over the first `keys` slots; returns [rows, vocab] logits (float32)."""
        valid = self.valid[:rows]
        valid.index_fill_(1, self.slot, True)
        mask = valid[:, :keys].view(rows, 1, 1, keys)
        h = F.embedding(self.tokens[:rows], self.embed).unsqueeze(1)

        def write_kv(i, k, v):
            self.k_cache[i, :rows].index_copy_(2, self.slot, k)
            self.v_cache[i, :rows].index_copy_(2, self.slot, v)

        def attend(i, q):
            return self._sdpa(q, self.k_cache[i, :rows, :, :keys], self.v_cache[i, :rows, :, :keys], mask)

        h = self._layers(h, self.positions[:rows].unsqueeze(1), write_kv, attend)
        return (h[:, 0] @ self.head.T).float()

    def _step(self, rows: int, keys: int):
        if not self.cuda_graphs:
            return self._decode(rows, keys)
        key = (rows, keys)
        if key not in self.graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm-up on a side stream, as CUDA graph capture requires
                self._decode(rows, keys)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self._decode(rows, keys)
            self.graphs[key] = (graph, out)
            self.stats["captures"] = self.stats.get("captures", 0) + 1
        graph, out = self.graphs[key]
        graph.replay()
        return out

    @torch.no_grad()
    def _prefill(self, ids, mask, n: int, rows: int):
        """Prefill each prompt once, copy its keys/values to its n rows; returns the next-token logits [P * n, vocab]."""
        p, width = ids.shape
        positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
        causal = torch.ones(width, width, dtype=torch.bool, device=self.device).tril()
        # a padding query attends to itself too, so no row is all -inf (its output is never read)
        attn_mask = (causal & mask.bool().view(p, 1, 1, width)) | torch.eye(width, dtype=torch.bool, device=self.device)
        h = F.embedding(ids, self.embed)

        def write_kv(i, k, v):
            self.k_cache[i, : p * n, :, :width] = k.repeat_interleave(n, dim=0)
            self.v_cache[i, : p * n, :, :width] = v.repeat_interleave(n, dim=0)
            self._kv = (k, v)

        def attend(i, q):
            k, v = self._kv
            return self._sdpa(q, k, v, attn_mask)

        h = self._layers(h, positions, write_kv, attend)
        self._kv = None
        last = mask.long().cumsum(-1).argmax(-1)  # the last real prompt token of each row
        logits = (h[torch.arange(p, device=self.device), last] @ self.head.T).float()
        self.valid[:rows].zero_()
        self.positions[:rows].zero_()
        self.valid[: p * n, :width] = mask.bool().repeat_interleave(n, dim=0)
        self.positions[: p * n] = mask.long().sum(-1).repeat_interleave(n)
        return logits.repeat_interleave(n, dim=0)

    @torch.no_grad()
    def generate(self, encoded: dict, n: int, temperature: float, max_tokens: int, pad_id: int, seed: int | None = None,
                 puzzles=None, think_budget: int = 0, return_logprobs: bool = False):
        """Sample `n` completions per prompt row; returns (sequences, finished_at) like VLLMRollout.generate."""
        if return_logprobs:
            raise ValueError("rollout_backend='native' does not return sampler log-probs (log_mismatch / tis_cap need vLLM)")
        if puzzles is not None and self.grammar is None:
            raise ValueError("puzzles need an engine built with constrain_answer=True")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        p, width = ids.shape
        b = p * n
        if width + max_tokens > self.max_len:
            raise ValueError(f"prompt width {width} + max_tokens {max_tokens} exceeds max_len {self.max_len}")
        rows = math.ceil(b / self.batch_bucket) * self.batch_bucket
        self._ensure_capacity(rows)
        generator = self.generator
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        constraints = None
        if puzzles is not None:
            constraints = [RequestConstraint(self.grammar, pz.numbers, pz.ops, self.index_cache, think_budget) for pz in puzzles for _ in range(n)]
        outputs: list[list[int]] = [[] for _ in range(b)]
        tails = [""] * b
        active = list(range(b))
        mask_cpu = 0.0
        allowed_log = [[] for _ in range(b)] if constraints is not None else None  # per row, per position (None = free)
        logits = self._prefill(ids, mask, n, rows)
        for s in range(max_tokens):
            if constraints is not None:
                start = time.perf_counter()
                allowed = [(r, constraints[r].allowed_ids(outputs[r])) for r in active]
                for r, row_allowed in allowed:
                    allowed_log[r].append(row_allowed or None)
                logits = mask_rows_cached(logits, allowed, self.index_cache)
                mask_cpu += time.perf_counter() - start
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                noise = torch.empty_like(probs).exponential_(generator=generator)
                sampled = (probs / noise).argmax(-1)
            else:
                sampled = logits.argmax(-1)
            chosen = sampled.tolist()
            still = []
            for r in active:
                tid = chosen[r]
                outputs[r].append(tid)
                text = tails[r] + (self.strings[tid] if tid < len(self.strings) else "")
                if tid in self.eos or CLOSE in text or len(outputs[r]) >= max_tokens:
                    continue
                tails[r] = text[1 - len(CLOSE) :]  # enough to see CLOSE complete across token boundaries
                still.append(r)
            active = still
            if not active:
                break
            self.tokens[:b].copy_(sampled)
            self.slot.fill_(width + s)
            keys = min(self.max_len, math.ceil((width + s + 1) / self.key_bucket) * self.key_bucket)
            logits = self._step(rows, keys)[:b]
            self.positions[:b] += 1
        self.mask_cpu = mask_cpu
        self.last_allowed = allowed_log  # what the grammar allowed at each sampled position (reuse_sampler_allowed)
        lengths = [len(o) for o in outputs]
        comp = torch.full((b, max(1, max(lengths))), pad_id, dtype=torch.long)
        for r, o in enumerate(outputs):
            comp[r, : len(o)] = torch.tensor(o, dtype=torch.long)
        sequences = torch.cat([ids.repeat_interleave(n, dim=0), comp.to(ids.device)], dim=1)
        return sequences, torch.tensor(lengths, dtype=torch.long, device=ids.device)
