"""Serving Mapika decider checkpoints (Mapika/decider-4b and its family) through `ask` and `serve`.

A Mapika decider checkpoint is a plain Qwen3.5 causal LM (`config.json`, `model.safetensors`, its
tokenizer) plus `decider_config.json`, which sets the prompt layout and the answer temperatures. It
has no `strands_decider_config.json`, no `serve_config.json` and no extra head: each answer is read
from the language model's own logits for the option-label tokens (A, B, C, ...) at the `(` that
ends the question's `Answer: (` line.

The rendering and the readout follow the release's own server (github.com/Mapika/decider,
Apache-2.0: `decider/systemone.py`, `decider/prompt.py`, `decider/prompt_fast.py`,
`decider/temperature.py`), in its default mode for `/v1/systemone` (independent questions,
isolated score levels, the plain state-first layout):

- every question is its own row: `Context:\\n<state>` then `\\n\\nQuestion: <text>\\nOptions:` with
  `\\n(A) <option>` lines and `\\nAnswer: (`; with more than 10 options each label is one token
  (A..Z, then two-letter labels);
- a choice option renders as `name` or `name: <description>`; a noul question as the options
  `no[: <false>]`, `yes[: <true>]`;
- a score question becomes one yes/no row per level ("Proposed answer: <level>"), and the levels'
  P(yes) are normalised into the score distribution;
- each slot's label logits are divided by the temperature of its answer type
  (`temperature_by_type`, else `temperature`; or `temperature_by_options`) and softmaxed.

The answers keep this package's shape: `derive_confidence` for choice (the same formula as the
release's), `derive_score_confidence` for score (the release uses another score formula and rounds
the score to 2 places). No yes/no floor is applied; decider-4b's own yes/no temperature (0.05)
makes its yes/no answers near 0 or 1.

Not supported: the chat layout (`"layout": "chat"`), the schema-first cache, and images.
"""

from __future__ import annotations

import json
import math
import os
import string
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from .modeling import CONFIG_NAME, LEGACY_CONFIG_NAME
from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Content,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)

DECIDER_CONFIG = "decider_config.json"
MAPIKA_FILES = ("*.json", "*.jinja", "model*.safetensors", "LICENSE")
LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)
MAX_OPTIONS = 255
DEFAULT_MAX_STATE_TOKENS = 32768
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
ANNOTATE_MIN = 8
ANSWER_TYPES = ("choice", "noul", "score")


# ------------------------------------------------------------------------------ rendering
def _txt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    """Long arrays get each element's position written in ({"_index": i, ...}), as the release does."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [
                {"_index": i, **annotate_indices(v, min_len)}
                if isinstance(v, dict)
                else {"_index": i, "value": annotate_indices(v, min_len)}
                for i, v in enumerate(x)
            ]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state: Content) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


@dataclass
class RenderedQuestion:
    kind: str  # "choice", "noul" or "score"
    text: str
    options: list[str]
    names: list[str]  # what the answer reports for each option (choice), else unused
    legend: list[str] = field(default_factory=list)


def render_question(question: Question) -> RenderedQuestion:
    raw: Any = question.instructions
    if isinstance(question, NoulQuestion):
        crit = question.criteria or {}
        false, true = crit.get("false"), crit.get("true")
        if raw in (None, ""):
            if false in (None, "") and true in (None, ""):
                raise ValueError(
                    "noul question without instructions: criteria must describe true or false"
                )
            text = NOUL_WITHOUT_INSTRUCTIONS
        else:
            text = _txt(raw)
        options = [
            "no" if false in (None, "") else f"no: {_txt(false)}",
            "yes" if true in (None, "") else f"yes: {_txt(true)}",
        ]
        return RenderedQuestion("noul", text, options, ["false", "true"])
    text = _txt(raw)
    if not text:
        raise ValueError("question without instructions")
    if isinstance(question, ChoiceQuestion):
        names = [str(n) for n in question.criteria]
        options = [
            n if d in (None, "") else f"{n}: {_txt(d)}"
            for n, d in zip(names, question.criteria.values(), strict=True)
        ]
        return RenderedQuestion("choice", text, options, names)
    legend = [_txt(c) for c in question.criteria]
    options = [f"{i}: {c}" for i, c in enumerate(legend)]
    return RenderedQuestion("score", text, options, [str(i) for i in range(len(legend))], legend)


def strip_level_number(text: str) -> str:
    import re

    return re.sub(r"^\s*-?\d+\s*:\s*", "", text)


def plan_rows(
    rendered: dict[str, RenderedQuestion],
) -> tuple[list[tuple[str, list[str], str]], list]:
    """One row (question text, options, answer type) per question; a score question becomes one
    yes/no row per level. -> (rows, index [(id, "iso" | "list", first row, n rows)])."""
    rows: list[tuple[str, list[str], str]] = []
    index: list[tuple[str, str, int, int]] = []
    for name, r in rendered.items():
        if r.kind == "score":
            level_rows = [
                (ISOLATED.format(q=r.text, level=strip_level_number(level)), ["no", "yes"], "score")
                for level in r.legend
            ]
            index.append((name, "iso", len(rows), len(level_rows)))
            rows += level_rows
        else:
            index.append((name, "list", len(rows), 1))
            rows.append((r.text, r.options, r.kind))
    return rows, index


# ------------------------------------------------------------------------------ tokens
class Labels:
    """The option-label tokens: A..J for up to 10 options (rendered in the question text), and the
    full table of MAX_OPTIONS single-token labels (A..Z, then two-letter ones) for wider questions."""

    def __init__(self, tok: Any) -> None:
        names = list(string.ascii_uppercase) + [
            a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
        ]
        ids: list[int] = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
            if len(ids) == MAX_OPTIONS:
                break
        if len(ids) != MAX_OPTIONS or len(set(ids)) != MAX_OPTIONS:
            raise ValueError("this tokenizer has no single-token label for every option")
        for j, letter in enumerate(LETTERS):
            if tok.encode(letter, add_special_tokens=False) != [ids[j]]:
                raise ValueError(f"label {letter!r} is not one token")
        self.ids = ids
        self.open_ids = tok.encode("\n(", add_special_tokens=False)


def question_piece(tok: Any, labels: Labels, text: str, options: Sequence[str]) -> list[int]:
    """`\\n\\nQuestion: <text>\\nOptions:` + options + `\\nAnswer: (` (one question per row)."""
    head = f"\n\nQuestion: {text}\nOptions:"
    tail = "\nAnswer: ("
    if len(options) <= NARROW:
        body = "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options))
        return list(tok.encode(head + body + tail, add_special_tokens=False))
    piece = list(tok.encode(head, add_special_tokens=False))
    for j, o in enumerate(options):
        piece += [
            *labels.open_ids,
            labels.ids[j],
            *list(tok.encode(f") {o}", add_special_tokens=False)),
        ]
    return piece + list(tok.encode(tail, add_special_tokens=False))


# ------------------------------------------------------------------------------ temperatures
@dataclass
class Temperatures:
    temperature: float = 1.0
    by_type: dict[str, float] = field(default_factory=dict)
    by_options: tuple[float, float, float] | None = None  # (a, b, min): T(n) = max(min, a + b ln n)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Temperatures:
        def positive(v: Any, where: str) -> float:
            if isinstance(v, bool):
                raise ValueError(f"{where} must be a finite number > 0")
            x = float(v)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{where} must be a finite number > 0, got {v!r}")
            return x

        t = positive(cfg.get("temperature", 1.0), "temperature")
        by_type = {}
        for k, v in (cfg.get("temperature_by_type") or {}).items():
            if k not in ANSWER_TYPES:
                raise ValueError(f"temperature_by_type: unknown answer type {k!r}")
            by_type[k] = positive(v, f"temperature_by_type[{k}]")
        spec = cfg.get("temperature_by_options")
        by_options = None
        if spec is not None:
            if by_type:
                raise ValueError(
                    '"temperature_by_options" and "temperature_by_type" cannot both be set'
                )
            by_options = (
                float(spec["a"]),
                float(spec["b"]),
                positive(spec.get("min", 0.05), "min"),
            )
        return cls(t, by_type, by_options)

    def of(self, kind: str, n_options: int) -> float:
        if self.by_options is not None:
            a, b, lo = self.by_options
            return max(lo, a + b * math.log(n_options))
        return self.by_type.get(kind, self.temperature)

    def report(self) -> dict[str, Any]:
        if self.by_options is not None:
            a, b, lo = self.by_options
            return {t: f"max({lo:g}, {a:g} + {b:g} ln n_options)" for t in ANSWER_TYPES}
        return {t: self.by_type.get(t, self.temperature) for t in ANSWER_TYPES}


# ------------------------------------------------------------------------------ detection
def _local_is_mapika(path: str) -> bool:
    has = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return has(DECIDER_CONFIG) and not has(CONFIG_NAME) and not has(LEGACY_CONFIG_NAME)


def is_mapika(checkpoint: str, revision: str | None = None) -> bool:
    """True when `checkpoint` (a local directory or a Hub repo id) has `decider_config.json` and no
    strands-decider config. For a Hub id only that file is fetched; any failure answers False."""
    if os.path.isdir(checkpoint):
        return _local_is_mapika(checkpoint)
    try:
        hf_hub_download(checkpoint, DECIDER_CONFIG, revision=revision)
    except Exception:
        return False
    for name in (CONFIG_NAME, LEGACY_CONFIG_NAME):
        try:
            hf_hub_download(checkpoint, name, revision=revision)
        except EntryNotFoundError:
            continue
        return False
    return True


def resolve(checkpoint: str, revision: str | None = None) -> str | None:
    """The local directory of a Mapika decider checkpoint, or None for any other checkpoint. The
    vision/ folder is not downloaded (images are not supported here)."""
    if not is_mapika(checkpoint, revision):
        return None
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(
            snapshot_download(checkpoint, revision=revision, allow_patterns=list(MAPIKA_FILES))
        )
    except Exception as e:
        raise FileNotFoundError(
            f"{checkpoint}: a Mapika decider checkpoint (it has {DECIDER_CONFIG}), but downloading "
            f"its files failed on the Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


def check_layout(cfg: dict[str, Any]) -> None:
    layout = cfg.get("layout") or ("chat" if cfg.get("chat_template") is True else "plain")
    if layout != "plain":
        raise ValueError(
            f"{DECIDER_CONFIG} names the prompt layout {layout!r}; only the plain layout is "
            "supported here (use the release's own server for this checkpoint)"
        )


# ------------------------------------------------------------------------------ the engine
@dataclass
class MapikaEngineConfig:
    device: str = "cuda"
    model_name: str = "decider-4b"
    # Not used by this engine (no state is cached); kept so /health reports the same keys.
    use_prefix_cache: bool = False
    max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS


def to_answer(r: RenderedQuestion, p: Sequence[float]) -> Answer:
    probs = [float(x) for x in p]
    total = sum(probs) or 1.0
    probs = [x / total for x in probs]
    if r.kind == "noul":
        return NoulAnswer(noul=round(probs[1], 4))
    if r.kind == "choice":
        best = max(range(len(probs)), key=probs.__getitem__)
        return ChoiceAnswer(
            choice=r.names[best],
            probabilities={n: round(x, 4) for n, x in zip(r.names, probs, strict=True)},
            confidence=round(derive_confidence(probs), 4),
        )
    return ScoreAnswer(
        score=round(sum(i * x for i, x in enumerate(probs)), 4),
        legend={str(i): d for i, d in enumerate(r.legend)},
        probabilities={str(i): round(x, 4) for i, x in enumerate(probs)},
        confidence=round(derive_score_confidence(probs), 4),
    )


class MapikaEngine:
    """Answers System One requests from a Mapika decider checkpoint. The same surface as
    `SystemOneEngine`: `evaluate(request)` and `ask(state, questions)`."""

    def __init__(
        self,
        lm: Any,
        tok: Any,
        decider_config: dict[str, Any],
        config: MapikaEngineConfig,
    ) -> None:
        check_layout(decider_config)
        self.lm = lm
        self.tok = tok
        self.decider_config = decider_config
        self.temperatures = Temperatures.from_config(decider_config)
        self.cfg = config
        self.labels = Labels(tok)
        weight = lm.get_output_embeddings().weight
        self.label_rows = weight[self.labels.ids].detach().clone()  # [MAX_OPTIONS, hidden]
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._tok_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        dev: torch.device = self.label_rows.device
        return dev

    def health_fields(self) -> dict[str, Any]:
        return {
            "base_model": str(
                self.decider_config.get("version") or self.decider_config.get("base")
            ),
            "max_length": self.cfg.max_state_tokens,
            "temperature": self.temperatures.report(),
        }

    def rows(self, request: SystemOneRequest) -> tuple[dict[str, RenderedQuestion], list, list]:
        rendered = {name: render_question(q) for name, q in request.questions.items()}
        plan, index = plan_rows(rendered)
        with self._tok_lock:
            ctx = list(
                self.tok.encode(
                    "Context:\n" + render_state(request.state), add_special_tokens=False
                )
            )[: self.cfg.max_state_tokens]
            items = []
            for text, options, kind in plan:
                if not 2 <= len(options) <= MAX_OPTIONS:
                    raise ValueError(f"2..{MAX_OPTIONS} options required")
                ids = ctx + question_piece(self.tok, self.labels, text, options)
                items.append((ids, len(options), kind))
        return rendered, index, items

    @torch.inference_mode()
    def probabilities(self, items: Sequence[tuple[list[int], int, str]]) -> list[list[float]]:
        """One forward over the right-padded rows; each row's label logits at its last token."""
        length = max(len(ids) for ids, _, _ in items)
        batch = torch.full((len(items), length), self.pad_id, dtype=torch.long)
        for b, (ids, _, _) in enumerate(items):
            batch[b, : len(ids)] = torch.tensor(ids)
        hidden = self.lm.model(input_ids=batch.to(self.device), use_cache=False).last_hidden_state
        last = torch.tensor([len(ids) - 1 for ids, _, _ in items], device=self.device)
        h = hidden[torch.arange(len(items), device=self.device), last]  # [N, hidden]
        logits = functional.linear(h, self.label_rows).float()  # [N, MAX_OPTIONS]
        out = []
        for row, (_, n, kind) in zip(logits, items, strict=True):
            t = self.temperatures.of(kind, n)
            out.append(torch.softmax(row[:n] / t, -1).cpu().tolist())
        return out

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError("images are not supported for a Mapika decider checkpoint here")
        rendered, index, items = self.rows(request)
        probs = self.probabilities(items)
        answers: dict[str, Answer] = {}
        for name, kind, start, n in index:
            r = rendered[name]
            if kind == "iso":
                fit = [probs[start + j][1] for j in range(n)]
                total = sum(fit) or 1e-9
                answers[name] = to_answer(r, [x / total for x in fit])
            else:
                answers[name] = to_answer(r, probs[start])
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=sum(len(ids) for ids, _, _ in items), output_tokens=0),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


def load_mapika_engine(
    path: str,
    *,
    device: str = "cuda",
    model_name: str = "decider-4b",
    attn_implementation: str | None = None,
) -> MapikaEngine:
    """A `MapikaEngine` from the local checkpoint directory `path` (see `resolve`)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "mlx":
        raise ValueError(
            "a Mapika decider checkpoint runs on torch devices (cuda, mps, cpu), not mlx"
        )
    with open(os.path.join(path, DECIDER_CONFIG), encoding="utf-8") as fh:
        decider_config = json.load(fh)
    check_layout(decider_config)
    extra: dict[str, Any] = {}
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    lm: Any = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **extra)
    lm.to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    max_state = int(decider_config.get("max_state_tokens", DEFAULT_MAX_STATE_TOKENS))
    return MapikaEngine(
        lm,
        tok,
        decider_config,
        MapikaEngineConfig(device=device, model_name=model_name, max_state_tokens=max_state),
    )
