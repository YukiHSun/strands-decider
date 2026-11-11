"""Serving OneJev `qev` checkpoints (baselabs/elephant-1) through `ask` and `serve`.

A qev checkpoint is a full fine-tune of a Qwen3.5 model (`config.json`, `model.safetensors`, its
tokenizer and chat template) plus `readout.json`, which names how the answer is read. elephant-1
also declares, under `modeling_overrides` in `config.json`, that its full-attention layers see
each block of the prompt bidirectionally. It has no `strands_decider_config.json`, no extra head
and no calibration file: each answer is the language model's own logits for the answer letters.

The rendering and the readout follow the runtime the release names (github.com/OmniJev/OneJev at
81ce62f with the release's `serve.patch`, Apache-2.0: `qev/prompt.py`, `qev/engine.py`,
`qev/readout.py`, `qev/answers.py`):

- every question is one chat: the system turn (`SYSTEM_PROMPT`), then a user turn of
  `<state>\\n<state text>\\n</state>\\n\\n` and the question
  (`Question: <text>\\n\\nOptions:\\nA. <option>...\\n\\nAnswer with one letter: A, B.`), rendered
  with the checkpoint's chat template (generation prompt, thinking off);
- a yes/no question has the options `A. yes: <true>`, `B. no: <false>`; a choice question
  `<letter>. <name>[: <description>]`; a score question `<letter>. level <i>: <level>`;
- the shared prefix is the longest common token prefix of the questions' prompts that ends at
  or before the end of the state block; with `noncausal` attention it is one block and every
  question's own tokens another: in the full-attention layers a token sees every token of its own
  block and every token of the blocks before it (the Gated DeltaNet layers stay causal);
- the answer is the LM head's logits (in float32) for the letters at the last token, divided by
  the calibration temperature (`calibration.json`; 1.0 without it) and softmaxed.

The release runs the prefix once and continues every question from its forked cache; this module
runs every question's prompt whole, under the same block mask, so it computes the same thing
(up to bf16 kernel rounding). The answers keep this package's shape: `derive_confidence` for
choice (the same formula as qev's), `derive_score_confidence` for score (qev uses another score
formula).

Not supported: the `head` readout, images, permutation debiasing (qev's default is off).
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import json
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
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

READOUT_CONFIG = "readout.json"
CALIBRATION = "calibration.json"
OVERRIDES_KEY = "modeling_overrides"
ELEPHANT_FILES = ("*.json", "*.jinja", "model*.safetensors", "LICENSE")

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK",
    "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
}  # fmt: skip
SLOT_LABELS: tuple[str, ...] = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)]
)
MAX_OPTIONS = 255
NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"
PREFIX_MARK = "QEV_PREFIX_MARK"
MASK_MARK = "strands_decider_block_mask"  # set on a modeling module once its mask is wrapped
MAX_BRANCH_TOKENS = 32768
MAX_REQUEST_TOKENS = 65536
ATTENTION = ("causal", "noncausal")
FULL_ATTENTION_MASK = {"causal": "causal", "block_bidirectional": "noncausal"}


# ------------------------------------------------------------------------------ rendering
def render_entry(entry: Any) -> str:
    """An instructions / criteria entry (string, object, array or null) as text."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Content | None) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


@dataclass
class RenderedQuestion:
    kind: str  # "choice", "noul" or "score"
    names: list[str]  # what the answer reports per option: choice names, else unused
    legend: list[str]  # score levels
    suffix: str
    n_slots: int


def _option_lines(names: Sequence[str], descriptions: Sequence[Any]) -> list[str]:
    lines = []
    for letter, name, desc in zip(SLOT_LABELS, names, descriptions, strict=False):
        text = render_entry(desc)
        lines.append(f"{letter}. {name}: {text}" if text else f"{letter}. {name}")
    return lines


def render_question(question: Question) -> RenderedQuestion:
    """One question's suffix (the text after the state block), as qev's `render_question` with
    the default (tagged) style and no option permutation."""
    instructions = render_entry(question.instructions)
    legend: list[str] = []
    if isinstance(question, NoulQuestion):
        kind = "noul"
        instructions = instructions or NOUL_DEFAULT_INSTRUCTIONS
        crit = question.criteria or {}
        true, false = crit.get("true"), crit.get("false")
        descs: list[Any] = [
            render_entry(true) if true is not None else NOUL_DEFAULT_TRUE,
            render_entry(false) if false is not None else NOUL_DEFAULT_FALSE,
        ]
        names = labels = ["yes", "no"]
    elif isinstance(question, ChoiceQuestion):
        kind = "choice"
        instructions = instructions or "Which option applies to the state?"
        names = labels = [str(n) for n in question.criteria]
        descs = list(question.criteria.values())
    else:
        kind = "score"
        instructions = instructions or "Which level describes the state?"
        legend = [str(c) for c in question.criteria]
        labels = [str(i) for i in range(len(legend))]
        names = [f"level {i}" for i in range(len(legend))]
        descs = list(legend)
    if not 1 <= len(labels) <= MAX_OPTIONS:
        raise ValueError(f"1..{MAX_OPTIONS} options required, got {len(labels)}")
    header = "Rate the state:" if kind == "score" else "Question:"
    letters = list(SLOT_LABELS[: len(labels)])
    answer_line = (
        "Answer with one letter: {}."
        if len(letters) <= len(LETTERS)
        else ("Answer with one label: {}.")
    )
    suffix = (
        f"{header} {instructions}\n\nOptions:\n"
        + "\n".join(_option_lines(names, descs))
        + "\n\n"
        + answer_line.format(", ".join(letters))
    )
    return RenderedQuestion(kind, labels if kind == "choice" else [], legend, suffix, len(labels))


def state_block(state: Content | None) -> str:
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def build_messages(state: Content | None, suffix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_block(state) + suffix},
    ]


# ------------------------------------------------------------------------------ settings
@dataclass
class ReadoutConfig:
    attention: str = "causal"
    readout: str = "slot_lm"

    @classmethod
    def load(cls, path: str) -> ReadoutConfig:
        """`readout.json`, with the attention taken from `config.json`'s `modeling_overrides` when
        it has them (as qev's `ReadoutConfig.load`). Only the `slot_lm` readout is supported."""
        with open(os.path.join(path, READOUT_CONFIG), encoding="utf-8") as fh:
            fields = json.load(fh)
        model_config = os.path.join(path, "config.json")
        overrides = None
        if os.path.exists(model_config):
            with open(model_config, encoding="utf-8") as fh:
                overrides = json.load(fh).get(OVERRIDES_KEY)
        attention = fields.get("attention", "causal")
        if overrides:
            mask = overrides["attention"]["full_attention"]
            if mask not in FULL_ATTENTION_MASK:
                raise ValueError(f"unsupported full_attention mask {mask!r} in {model_config}")
            if "attention" in fields and fields["attention"] != FULL_ATTENTION_MASK[mask]:
                raise ValueError(f"{READOUT_CONFIG} and config.json disagree on the attention")
            attention = FULL_ATTENTION_MASK[mask]
        if attention not in ATTENTION:
            raise ValueError(f"attention must be one of {ATTENTION}, got {attention!r}")
        readout = fields.get("readout", "slot_lm")
        if readout != "slot_lm":
            raise ValueError(
                f"{READOUT_CONFIG} names the {readout!r} readout; only slot_lm is supported here"
            )
        return cls(attention=attention, readout=readout)


@dataclass
class Calibration:
    """Temperatures per "kind" or "kind:K" (qev's `Calibration`); missing keys fall back to the
    kind's temperature, then to 1.0."""

    temperatures: dict[str, float]

    @classmethod
    def load(cls, path: str) -> Calibration:
        file = os.path.join(path, CALIBRATION)
        if not os.path.exists(file):
            return cls({})
        with open(file, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls({k: float(v) for k, v in data.get("temperatures", {}).items()})

    def of(self, kind: str, k: int) -> float:
        return self.temperatures.get(f"{kind}:{k}", self.temperatures.get(kind, 1.0))


# ------------------------------------------------------------------------------ block mask
_BLOCKS: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar(
    "elephant_blocks", default=None
)


def install_block_mask(lm: Any) -> None:
    """Route block ids (set with `blocks`) into the full-attention mask of `lm`'s modeling module,
    as qev's `install_noncausal`: transformers' `create_causal_mask(block_sequence_ids=...)` lets a
    token see its own block both ways. Outside a `blocks` context the model is unchanged."""
    impl = getattr(lm.config, "_attn_implementation", None)
    if impl not in ("sdpa", "eager", "flex_attention"):
        raise ValueError(f"block attention needs sdpa, eager or flex_attention, not {impl}")
    mod: Any = importlib.import_module(type(lm).__module__)
    if getattr(mod, MASK_MARK, False):
        return
    if not hasattr(mod, "create_causal_mask"):
        raise ValueError(f"{mod.__name__} builds no causal mask to extend")
    original: Callable[..., Any] = mod.create_causal_mask

    def create_causal_mask(*args: Any, **kwargs: Any) -> Any:
        ids = _BLOCKS.get()
        if ids is not None and kwargs.get("block_sequence_ids") is None:
            kwargs["block_sequence_ids"] = ids
        return original(*args, **kwargs)

    mod.create_causal_mask = create_causal_mask
    setattr(mod, MASK_MARK, True)


@contextlib.contextmanager
def blocks(ids: torch.Tensor | None) -> Iterator[None]:
    token = _BLOCKS.set(ids)
    try:
        yield
    finally:
        _BLOCKS.reset(token)


def block_ids(prefix: int, lengths: Sequence[int], width: int) -> torch.Tensor:
    """0 on the shared prefix, 1 on each row's own tokens, -1 on right padding (causal)."""
    pos = torch.arange(width)[None]
    n = torch.tensor(list(lengths))[:, None]
    return torch.where(pos < prefix, 0, torch.where(pos < n, 1, -1))


# ------------------------------------------------------------------------------ detection
def _local_is_elephant(path: str) -> bool:
    has = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return has(READOUT_CONFIG) and not has(CONFIG_NAME) and not has(LEGACY_CONFIG_NAME)


def is_elephant(checkpoint: str, revision: str | None = None) -> bool:
    """True when `checkpoint` (a local directory or a Hub repo id) has `readout.json` and no
    strands-decider config. For a Hub id only that file is fetched; any failure answers False."""
    if os.path.isdir(checkpoint):
        return _local_is_elephant(checkpoint)
    try:
        hf_hub_download(checkpoint, READOUT_CONFIG, revision=revision)
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
    """The local directory of a qev checkpoint, or None for any other checkpoint. Only the
    weights, configs and tokenizer are downloaded: the release's own scripts (engine.py,
    setup.sh, serve.patch) are not."""
    if not is_elephant(checkpoint, revision):
        return None
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(
            snapshot_download(checkpoint, revision=revision, allow_patterns=list(ELEPHANT_FILES))
        )
    except Exception as e:
        raise FileNotFoundError(
            f"{checkpoint}: a qev checkpoint (it has {READOUT_CONFIG}), but downloading its files "
            f"failed on the Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


# ------------------------------------------------------------------------------ the engine
@dataclass
class ElephantEngineConfig:
    device: str = "cuda"
    model_name: str = "elephant-1"
    # Not used by this engine (no state is cached); kept so /health reports the same keys.
    use_prefix_cache: bool = False


def to_answer(r: RenderedQuestion, p: Sequence[float]) -> Answer:
    probs = [float(x) for x in p]
    total = sum(probs) or 1.0
    probs = [x / total for x in probs]
    if r.kind == "noul":
        return NoulAnswer(noul=round(probs[0], 4))  # option A is yes
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


def slot_token_ids(tok: Any) -> list[int]:
    """The token id of every answer label that is one round-trip token (qev's `_verify_slots`):
    A..Z are required; the two-letter labels are all or nothing."""

    def one(label: str) -> int | None:
        enc = tok.encode(label, add_special_tokens=False)
        return int(enc[0]) if len(enc) == 1 and tok.decode(enc) == label else None

    ids = [one(c) for c in LETTERS]
    if any(i is None for i in ids):
        raise ValueError("the answer letters A..Z are not single tokens for this tokenizer")
    extra = [one(label) for label in SLOT_LABELS[len(LETTERS) :]]
    out = [int(i) for i in ids if i is not None]
    if all(i is not None for i in extra):
        out += [int(i) for i in extra if i is not None]
    if len(set(out)) != len(out):
        raise ValueError("answer slot tokens collide")
    return out


class ElephantEngine:
    """Answers System One requests from a qev checkpoint. The same surface as
    `SystemOneEngine`: `evaluate(request)` and `ask(state, questions)`."""

    def __init__(
        self,
        lm: Any,
        tok: Any,
        readout: ReadoutConfig,
        calibration: Calibration,
        config: ElephantEngineConfig,
        *,
        template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.lm = lm
        self.tok = tok
        self.readout = readout
        self.calibration = calibration
        self.cfg = config
        self.template_kwargs = template_kwargs if template_kwargs is not None else {}
        self.slot_ids = slot_token_ids(tok)
        weight = lm.get_output_embeddings().weight
        # qev computes the LM head in float32; only the slot rows are ever read.
        self.slot_rows = weight[self.slot_ids].detach().to(torch.float32).clone()
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._tok_lock = threading.Lock()
        if readout.attention == "noncausal":
            install_block_mask(lm)

    @property
    def device(self) -> torch.device:
        dev: torch.device = self.slot_rows.device
        return dev

    def health_fields(self) -> dict[str, Any]:
        return {
            "attention": self.readout.attention,
            "readout": self.readout.readout,
            "max_length": MAX_BRANCH_TOKENS,
            "temperature": self.calibration.temperatures or 1.0,
        }

    def _prompt(self, state: Content | None, suffix: str) -> str:
        text: str = self.tok.apply_chat_template(
            build_messages(state, suffix),
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )
        return text

    def rows(
        self, request: SystemOneRequest
    ) -> tuple[dict[str, RenderedQuestion], list[list[int]], int]:
        """Each question's full prompt ids and the shared prefix length (qev's
        `_prefix_and_suffix_ids`: the longest common token prefix that ends at or before the end of
        the state block)."""
        rendered = {name: render_question(q) for name, q in request.questions.items()}
        if any(r.n_slots > len(self.slot_ids) for r in rendered.values()):
            raise ValueError(f"at most {len(self.slot_ids)} options for this tokenizer")
        with self._tok_lock:
            full = [
                list(
                    self.tok.encode(self._prompt(request.state, r.suffix), add_special_tokens=False)
                )
                for r in rendered.values()
            ]
            marked = self._prompt(request.state, PREFIX_MARK)
            prefix = list(
                self.tok.encode(marked[: marked.index(PREFIX_MARK)], add_special_tokens=False)
            )
        n = len(prefix)
        while n > 0 and not all(ids[:n] == prefix[:n] for ids in full):
            n -= 1
        if any(len(ids) <= n for ids in full):
            raise ValueError("a question has no tokens after the state block")
        longest = max(len(ids) for ids in full)
        total = n + sum(len(ids) - n for ids in full)
        if longest > MAX_BRANCH_TOKENS:
            raise ValueError(
                f"state plus longest question is {longest} tokens, limit is {MAX_BRANCH_TOKENS}"
            )
        if total > MAX_REQUEST_TOKENS:
            raise ValueError(f"request is {total} tokens, limit is {MAX_REQUEST_TOKENS}")
        return rendered, full, n

    @torch.inference_mode()
    def slot_logits(self, rows: Sequence[list[int]], prefix: int) -> torch.Tensor:
        """One forward over the right-padded rows under the block mask; each row's float32 slot
        logits at its last token. [N, slots]"""
        width = max(len(ids) for ids in rows)
        batch = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for b, ids in enumerate(rows):
            batch[b, : len(ids)] = torch.tensor(ids)
            mask[b, : len(ids)] = 1
        ids_on = None
        if self.readout.attention == "noncausal":
            ids_on = block_ids(prefix, [len(ids) for ids in rows], width).to(self.device)
        with blocks(ids_on):
            hidden = self.lm.model(
                input_ids=batch.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            ).last_hidden_state
        last = torch.tensor([len(ids) - 1 for ids in rows], device=self.device)
        h = hidden[torch.arange(len(rows), device=self.device), last].to(torch.float32)
        out: torch.Tensor = functional.linear(h, self.slot_rows)
        return out

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError("images are not supported for a qev checkpoint here")
        rendered, rows, prefix = self.rows(request)
        logits = self.slot_logits(rows, prefix).cpu()
        answers: dict[str, Answer] = {}
        for (name, r), row in zip(rendered.items(), logits, strict=True):
            t = self.calibration.of(r.kind, r.n_slots)
            p = torch.softmax(row[: r.n_slots] / t, -1).tolist()
            answers[name] = to_answer(r, p)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(
                input_tokens=prefix + sum(len(ids) - prefix for ids in rows), output_tokens=0
            ),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


def load_elephant_engine(
    path: str,
    *,
    device: str = "cuda",
    model_name: str = "elephant-1",
    attn_implementation: str | None = None,
) -> ElephantEngine:
    """An `ElephantEngine` from the local checkpoint directory `path` (see `resolve`). The text
    model is loaded (the vision tower's weights are not), as qev does."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if device == "mlx":
        raise ValueError("a qev checkpoint runs on torch devices (cuda, mps, cpu), not mlx")
    readout = ReadoutConfig.load(path)
    config = AutoConfig.from_pretrained(path)
    cls: Any = AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from transformers import Qwen3_5ForCausalLM

        cls = Qwen3_5ForCausalLM
        config = config.get_text_config()
    extra: dict[str, Any] = {}
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    lm: Any = cls.from_pretrained(path, config=config, dtype=dtype, **extra)
    lm.to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    template_kwargs = {"enable_thinking": False} if "qwen" in config.model_type else {}
    return ElephantEngine(
        lm,
        tok,
        readout,
        Calibration.load(path),
        ElephantEngineConfig(device=device, model_name=model_name),
        template_kwargs=template_kwargs,
    )
