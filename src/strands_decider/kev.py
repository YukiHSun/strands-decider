"""Serving Kev checkpoints (jaredpalmer/kev-4b and its family) through `ask` and `serve`.

A Kev checkpoint is a LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) on a
Qwen base named in its `head.pt`, or a full backbone, plus `head.pt`: the pointer head's weights, its
calibration temperature and the base it was trained on (`base`, `base_revision`). It has no
`strands_decider_config.json`. Each answer is read by the pointer head from the backbone's hidden
states at the question's `<decide>` token and at the end of each option.

The rendering and the readout follow the release's own code (github.com/jaredpalmer/kev,
Apache-2.0: `kev/api.py`, `kev/model.py`, `kev/checkpoint.py`) in the path its server takes on a
Qwen3.5 (hybrid) backbone:

- the state and every question are rendered to text as `kev.api` does (`render`, `option_text`;
  noul as the options `no[: <false>]`, `yes[: <true>]`; score levels as given);
- the record is packed with Qwen's rarely used special tokens as delimiters:
  `<|fim_prefix|>` state, then per question `<|fim_middle|>` instruction,
  `<|box_start|>` option `<|box_end|>` ..., `<|fim_suffix|>` (caller text can never produce them);
- every question runs as its own causal row (the state, then its branch, positions continuing);
- logits = q(h_decide) . k(h_option_end) / sqrt(head_dim), divided by head.pt's temperature,
  softmaxed over the question's options.

The LoRA adapter is merged into the base weights at load time (delta computed in fp32, one rounding
to the load dtype), as the release does by default. No yes/no floor is applied.

Not supported: option isolation, adapters with trained token embeddings, attention-only bases
(the packed block-causal form), and images.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
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

HEAD_FILE = "head.pt"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
KEV_FILES = ("*.json", "*.safetensors", HEAD_FILE, "*.txt", "*.jinja")
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
SERVE_MAX_STATE = 65536
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


# ------------------------------------------------------------------------------ rendering
def render(v: Any, indent: int = 0) -> str:
    """str | object | array as the text the model sees (kev.api.render): field names kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}"
        if isinstance(x, (dict, list))
        else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: Any) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


@dataclass
class KevQuestion:
    kind: str
    instruction: str
    options: list[str]
    keys: list[str]
    legend: list[str]


def to_question(question: Question) -> KevQuestion:
    if isinstance(question, NoulQuestion):
        c = question.criteria or {}
        options = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        return KevQuestion("noul", render(question.instructions), options, ["false", "true"], [])
    if isinstance(question, ChoiceQuestion):
        keys = [str(k) for k in question.criteria]
        options = [option_text(k, v) for k, v in zip(keys, question.criteria.values(), strict=True)]
        return KevQuestion("choice", render(question.instructions), options, keys, [])
    options = [render(x) for x in question.criteria]
    keys = [str(i) for i in range(len(options))]
    return KevQuestion("score", render(question.instructions), options, keys, options)


def user_tokens(tok: Any, text: str) -> list[int]:
    """Caller text tokenised so it can never produce a delimiter token: `<|name|>` becomes `<¦name¦>`."""
    return list(tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids)


@dataclass
class Row:
    ids: list[int]
    decide: int
    option_ends: list[int]


def encode_rows(
    tok: Any, state: str, questions: Sequence[KevQuestion], max_state: int = SERVE_MAX_STATE
) -> tuple[list[Row], int]:
    """One causal row per question: the state, then that question's branch. -> (rows, state tokens)."""
    s_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    state_tokens = user_tokens(tok, state)
    if len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    head = [s_id, *state_tokens]
    rows = []
    for q in questions:
        branch = [q_id, *user_tokens(tok, q.instruction)]
        ends = []
        for option in q.options:
            branch += [o_id, *user_tokens(tok, option), c_id]
            ends.append(len(head) + len(branch) - 1)
        branch.append(d_id)
        rows.append(Row(head + branch, len(head) + len(branch) - 1, ends))
    return rows, len(head)


# ------------------------------------------------------------------------------ head
class PointerHead(torch.nn.Module):
    def __init__(self, d: int, dp: int = 256) -> None:
        super().__init__()
        self.q, self.k = torch.nn.Linear(d, dp), torch.nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return out


# ------------------------------------------------------------------------------ detection
def _local_is_kev(path: str) -> bool:
    has = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return has(HEAD_FILE) and not has(CONFIG_NAME) and not has(LEGACY_CONFIG_NAME)


def is_kev(checkpoint: str, revision: str | None = None) -> bool:
    """True when `checkpoint` (a local directory or a Hub repo id) has `head.pt` and no
    strands-decider config. For a Hub id only `head.pt` (5 MB) is fetched; any failure answers False."""
    if os.path.isdir(checkpoint):
        return _local_is_kev(checkpoint)
    try:
        hf_hub_download(checkpoint, HEAD_FILE, revision=revision)
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
    """The local directory of a Kev checkpoint, or None for any other checkpoint."""
    if not is_kev(checkpoint, revision):
        return None
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(snapshot_download(checkpoint, revision=revision, allow_patterns=list(KEV_FILES)))
    except Exception as e:
        raise FileNotFoundError(
            f"{checkpoint}: a Kev checkpoint (it has {HEAD_FILE}), but downloading its files failed "
            f"on the Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


def read_meta(path: str) -> dict[str, Any]:
    """head.pt: tensors and plain values only (weights_only=True: no code is unpickled)."""
    meta = torch.load(os.path.join(path, HEAD_FILE), map_location="cpu", weights_only=True)
    if not isinstance(meta, dict) or "head" not in meta or "base" not in meta:
        raise ValueError(f"{path}/{HEAD_FILE} is not a Kev head file")
    return meta


# ------------------------------------------------------------------------------ the engine
@dataclass
class KevEngineConfig:
    device: str = "cuda"
    model_name: str = "kev-4b"
    use_prefix_cache: bool = False
    max_state_tokens: int = SERVE_MAX_STATE


def to_answer(q: KevQuestion, probs: Sequence[float]) -> Answer:
    p = [float(x) for x in probs]
    if q.kind == "noul":
        return NoulAnswer(noul=round(p[1], 4))
    if q.kind == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return ChoiceAnswer(
            choice=q.keys[best],
            probabilities={k: round(x, 4) for k, x in zip(q.keys, p, strict=True)},
            confidence=round(derive_confidence(p), 4),
        )
    return ScoreAnswer(
        score=round(sum(i * x for i, x in enumerate(p)), 4),
        legend=dict(zip(q.keys, q.legend, strict=True)),
        probabilities={k: round(x, 4) for k, x in zip(q.keys, p, strict=True)},
        confidence=round(derive_score_confidence(p), 4),
    )


class KevEngine:
    """Answers System One requests from a Kev checkpoint. The same surface as `SystemOneEngine`."""

    def __init__(
        self,
        backbone: Any,
        head: PointerHead,
        tok: Any,
        temperature: float,
        config: KevEngineConfig,
        *,
        base_model: str = "",
    ) -> None:
        self.backbone = backbone
        self.head = head
        self.tok = tok
        self.temperature = float(temperature)
        self.cfg = config
        self.base_model = base_model
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._tok_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        dev: torch.device = next(self.head.parameters()).device
        return dev

    def health_fields(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "max_length": self.cfg.max_state_tokens,
            "temperature": self.temperature,
        }

    @torch.inference_mode()
    def probabilities(self, rows: Sequence[Row]) -> list[list[float]]:
        length = max(len(r.ids) for r in rows)
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), length), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, : len(r.ids)] = torch.tensor(r.ids)
            mask[b, : len(r.ids)] = 1
        positions = torch.arange(length).expand(len(rows), length)
        hidden = self.backbone(
            input_ids=ids.to(self.device),
            position_ids=positions.to(self.device),
            attention_mask=mask.to(self.device),
        ).last_hidden_state.float()
        out = []
        for b, r in enumerate(rows):
            h = hidden[b]
            z = self.head(h[r.decide], h[torch.tensor(r.option_ends, device=self.device)])
            out.append(torch.softmax(z / self.temperature, -1).cpu().tolist())
        return out

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError("images are not supported for a Kev checkpoint")
        questions = {name: to_question(q) for name, q in request.questions.items()}
        with self._tok_lock:
            rows, _ = encode_rows(
                self.tok, render(request.state), list(questions.values()), self.cfg.max_state_tokens
            )
        probs = self.probabilities(rows)
        answers = {
            name: to_answer(q, p) for (name, q), p in zip(questions.items(), probs, strict=True)
        }
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=sum(len(r.ids) for r in rows), output_tokens=0),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


# ------------------------------------------------------------------------------ loading
LORA_TARGETS_HINT = "lora_A.weight"


def merge_lora(backbone: Any, path: str) -> int:
    """Fold the adapter into `backbone` (the base's text model): W += (alpha / r) * B @ A, computed
    in fp32 and rounded once to the weight's dtype. Returns the number of merged modules."""
    from safetensors.torch import load_file

    with open(os.path.join(path, ADAPTER_CONFIG), encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg.get("trainable_token_indices") or cfg.get("modules_to_save"):
        raise ValueError(
            "Kev adapters with trained token embeddings or saved modules are not supported"
        )
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise ValueError("DoRA / rsLoRA adapters are not supported")
    r = int(cfg["r"])
    scale = float(cfg["lora_alpha"]) / r
    weights = load_file(os.path.join(path, ADAPTER_WEIGHTS))
    modules = dict(backbone.named_modules())
    merged = 0
    for key, a in weights.items():
        if not key.endswith(LORA_TARGETS_HINT):
            continue
        name = key[: -len(".lora_A.weight")]
        for prefix in ("base_model.model.model.", "base_model.model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        b = weights[key.replace("lora_A", "lora_B")]
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"adapter module {name!r} is not in the base model")
        with torch.no_grad():
            w = module.weight
            delta = (b.float() @ a.float()) * scale
            module.weight.copy_((w.float() + delta.to(w.device)).to(w.dtype))
        merged += 1
    if merged == 0:
        raise ValueError(f"{path}: the adapter has no LoRA weights")
    return merged


def load_kev_engine(
    path: str,
    *,
    device: str = "cuda",
    model_name: str = "kev-4b",
    attn_implementation: str | None = None,
) -> KevEngine:
    """A `KevEngine` from the local Kev checkpoint directory `path` (see `resolve`). The base the
    adapter names in head.pt (`base` at `base_revision`) is loaded from the Hugging Face Hub."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if device == "mlx":
        raise ValueError("a Kev checkpoint runs on torch devices (cuda, mps, cpu) here, not mlx")
    meta = read_meta(path)
    if meta.get("option_isolation"):
        raise ValueError("Kev checkpoints with option_isolation are not supported")
    if meta.get("special_embeddings"):
        raise ValueError("Kev checkpoints with trained special-token embeddings are not supported")
    full = meta.get("weights", "lora") == "full"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    load: dict[str, Any] = {"dtype": dtype}
    if attn_implementation is not None:
        load["attn_implementation"] = attn_implementation
    if full:
        backbone: Any = AutoModel.from_pretrained(path, **load)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            meta["base"], revision=meta.get("base_revision"), **load
        ).model
    text_config = backbone.config.get_text_config()
    if "linear_attention" not in set(getattr(text_config, "layer_types", None) or []):
        raise ValueError(
            "only Kev checkpoints on hybrid (Qwen3.5) bases are supported here; this base is "
            "attention-only (it needs the packed block-causal form)"
        )
    if not full:
        merge_lora(backbone, path)
    backbone.to(device).eval()
    head = PointerHead(text_config.hidden_size, dp=int(meta.get("head_dim", 256)))
    head.load_state_dict(meta["head"])
    head.to(device=device, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(meta["base"], revision=meta.get("base_revision"))
    return KevEngine(
        backbone,
        head,
        tok,
        float(meta.get("temperature", 1.0)),
        KevEngineConfig(device=device, model_name=model_name),
        base_model=f"{meta['base']}@{meta.get('base_revision') or 'main'}",
    )
