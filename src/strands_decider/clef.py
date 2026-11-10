"""Serving Cloudflare Clef checkpoints (Cloudflare/clef-flash) through `ask` and `serve`.

A Clef checkpoint is a merged Qwen3.5 vision-language backbone (`config.json`, sharded
`model-*.safetensors`, its tokenizer and processor) plus a joint schema head
(`joint_head_config.json`, `joint_head.safetensors`). It has no `strands_decider_config.json`
and no `serve_config.json`. The head reads the backbone's last hidden states once per request
and returns one logit per allowed option for every question at once; nothing is generated.

The prompt layout, the head and the option ordering below follow the release's reference
implementation (`joint_schema_model.py` in the Cloudflare/clef-flash repo, Apache-2.0), because
the head was trained on exactly that layout. What differs from the reference is only the answer
shape, which follows the other engines of this package:

- choice: `confidence` is `derive_confidence` (normalised max-probability: uniform -> 0,
  one-hot -> 1), not the raw max-probability the reference returns; the raw values are in
  `probabilities`.
- score: `confidence` is `derive_score_confidence`.
- noul: P(true), as in the reference. There is no floor.

Images (base64 or data: URIs, `images` in the request) are always accepted: the release has its
vision tower merged in. Each image is shrunk to at most `MAX_PIXELS` pixels before the processor.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
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

HEAD_CONFIG = "joint_head_config.json"
HEAD_WEIGHTS = "joint_head.safetensors"
# The files a Clef checkpoint is read from (the repo's evaluation results are not downloaded).
CLEF_FILES = (
    "*.json",
    "*.jinja",
    "model-*.safetensors",
    HEAD_WEIGHTS,
    "LICENSE",
)
# The reference's default prompt window.
DEFAULT_MAX_LENGTH = 16384
# Each image is shrunk (never enlarged) to at most this many pixels before the processor; the
# release's processor itself allows up to 16M pixels, which would not fit a request window.
MAX_PIXELS = 1024 * 1024

# ----------------------------------------------------------------------------- reference layout
# From the release's joint_schema_model.py (Apache-2.0), unchanged.
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
MEDIA_BATCH_KEYS = ("pixel_values", "image_grid_thw")
MEDIA_TOKEN_KEYS = ("mm_token_type_ids",)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
_NOUL_DEFAULTS = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


def render(value: Any) -> str:
    """A state, an instruction or an option description as the reference renders it: a string
    as is, anything else as compact JSON with sorted keys."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: Question) -> list[tuple[str, Any]]:
    """The (option id, description) pairs of a question, in the order the prompt lists them:
    noul as true then false (descriptions default to the reference's), choice sorted by option
    id, score by level index."""
    if isinstance(question, NoulQuestion):
        criteria: dict[str, Any] = dict(_NOUL_DEFAULTS)
        # A null description keeps the default one (the request's JSON would omit it).
        criteria.update({k: v for k, v in (question.criteria or {}).items() if v is not None})
        return [(key, criteria[key]) for key in ("true", "false")]
    if isinstance(question, ChoiceQuestion):
        return sorted((str(key), value) for key, value in question.criteria.items())
    return [(str(index), value) for index, value in enumerate(question.criteria)]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    media: dict[str, Any] | None = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def _encode_media(processor: Any, images: Sequence[Any]) -> tuple[list[int], dict[str, Any] | None]:
    if not images:
        return [], None
    if processor is None:
        raise ValueError("records with images require a processor")
    text = IMAGE_PLACEHOLDER * len(images) + "\n"
    encoded = processor(text=[text], images=list(images), return_tensors="pt")
    media = {key: encoded[key] for key in MEDIA_BATCH_KEYS if key in encoded}
    for key in MEDIA_TOKEN_KEYS:
        if key in encoded:
            media[key] = encoded[key][0].tolist()
    return list(encoded["input_ids"][0].tolist()), media


def encode_record(
    tokenizer: Any,
    state: Content,
    questions: dict[str, Question],
    *,
    images: Sequence[Any] = (),
    processor: Any = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> EncodedRecord:
    """The reference's prompt for one request, with the token spans the head reads.

    As in the reference, the schema is laid out first and the state is truncated to whatever of
    the window the schema leaves; a schema that alone exceeds the window is refused.
    """
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    encoded_questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(questions.items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question.type}\n"
                "INSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions: Any = question.instructions
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        encoded_questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[question.type],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, images)
    if media is not None:
        media["token_offset"] = len(prefix_ids)
        prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(state))
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id=q.question_id,
            question_type=q.question_type,
            question_span=(q.question_span[0] + offset, q.question_span[1] + offset),
            option_spans=tuple((a + offset, b + offset) for a, b in q.option_spans),
            option_ids=q.option_ids,
        )
        for q in encoded_questions
    )
    return EncodedRecord(
        input_ids=tuple(prefix_ids + state_ids + schema_ids + suffix_ids),
        questions=shifted,
        media=media,
    )


def collate(record: EncodedRecord, device: torch.device) -> dict[str, Any]:
    """One record as the backbone's inputs (no padding: a request is one record)."""
    length = len(record.input_ids)
    batch: dict[str, Any] = {
        "input_ids": torch.tensor([record.input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, length), dtype=torch.long, device=device),
        "records": [record],
        "media": {},
    }
    if record.media:
        media: dict[str, torch.Tensor] = {
            key: record.media[key].to(device) for key in MEDIA_BATCH_KEYS if key in record.media
        }
        for key in MEDIA_TOKEN_KEYS:
            if key in record.media:
                values = torch.zeros((1, length), dtype=torch.long, device=device)
                offset = record.media["token_offset"]
                ids = torch.tensor(record.media[key], dtype=torch.long, device=device)
                values[0, offset : offset + len(ids)] = ids
                media[key] = values
        batch["media"] = media
    return batch


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(queries)
        memory = self.memory_norm(memory)
        routed, _ = self.attention(normalized, memory, memory, need_weights=False)
        queries = queries + self.attention_dropout(routed)
        out: torch.Tensor = queries + self.feedforward(self.feedforward_norm(queries))
        return out


class JointSchemaHead(torch.nn.Module):
    """The release's joint schema head (joint_schema_model.py), unchanged."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width, heads=heads, feedforward=feedforward, dropout=dropout
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [self._mean_span(sequence_hidden, q.question_span) for q in record.questions]
            )
            type_ids = torch.tensor(
                [q.question_type for q in record.questions], device=hidden_states.device
            )
            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []
            for q in record.questions:
                context_vectors = torch.stack(
                    [self._mean_span(sequence_hidden, span) for span in q.option_spans]
                )
                lexical = torch.stack(
                    [
                        output_embedding_weight[input_ids[batch_index, start:end]].mean(dim=0)
                        for start, end in q.option_spans
                    ]
                )
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(q.option_spans))

            option_queries = [
                self.option_context_projection(context_vectors)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[i]).unsqueeze(0)
                for i, (context_vectors, lexical) in enumerate(
                    zip(option_contexts, lexical_options, strict=True)
                )
            ]
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field_vector, options in zip(base_fields, split_options, strict=True):
                weights = torch.softmax(
                    torch.matmul(options, field_vector) / math.sqrt(options.shape[-1]), dim=0
                )
                option_summaries.append(torch.sum(weights.unsqueeze(-1) * options, dim=0))
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field_vector, lexical, routed in zip(
                fields, lexical_options, split_options, strict=True
            ):
                anchor = functional.normalize(
                    question_vectors[len(record_logits)] + global_vector, dim=-1
                )
                lexical_anchor = functional.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)
                options = self.option_norm(routed)
                repeated = field_vector.unsqueeze(0).expand_as(options)
                cosine = functional.cosine_similarity(repeated, options, dim=-1)
                features = torch.cat(
                    [repeated, options, repeated * options, torch.abs(repeated - options)],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
            results.append(record_logits)
        return results


class ClefModel(torch.nn.Module):
    """The backbone and the head; a text-only request runs the text model alone."""

    def __init__(self, language_model: Any, head: JointSchemaHead) -> None:
        super().__init__()
        self.language_model = language_model
        self.head = head

    def forward(self, batch: dict[str, Any]) -> list[list[torch.Tensor]]:
        media = batch.get("media") or {}
        text_model = self.language_model.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        outputs = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        result: list[list[torch.Tensor]] = self.head(
            outputs.last_hidden_state,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            self.language_model.get_output_embeddings().weight,
        )
        return result


# ------------------------------------------------------------------------------ detection
def _local_is_clef(path: str) -> bool:
    has = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return (
        has(HEAD_CONFIG)
        and has(HEAD_WEIGHTS)
        and not has(CONFIG_NAME)
        and not has(LEGACY_CONFIG_NAME)
    )


def is_clef(checkpoint: str, revision: str | None = None) -> bool:
    """True when `checkpoint` (a local directory or a Hub repo id) is a Clef checkpoint: it has
    `joint_head_config.json` (and, locally, `joint_head.safetensors`) and no decider config.

    For a Hub id only `joint_head_config.json` is fetched. Any failure to fetch it (no such file,
    no such repo, offline, not a repo id) answers False, so the other loaders run as before.
    """
    if os.path.isdir(checkpoint):
        return _local_is_clef(checkpoint)
    try:
        hf_hub_download(checkpoint, HEAD_CONFIG, revision=revision)
    except Exception:  # EntryNotFoundError and any other failure: not ours to report
        return False
    for name in (CONFIG_NAME, LEGACY_CONFIG_NAME):
        try:
            hf_hub_download(checkpoint, name, revision=revision)
        except EntryNotFoundError:
            continue
        return False  # a decider checkpoint that also ships a head config is not Clef's
    return True


def resolve(checkpoint: str, revision: str | None = None) -> str | None:
    """The local directory of a Clef checkpoint, or None for any other checkpoint. A Hub repo is
    downloaded with `allow_patterns` limited to `CLEF_FILES` at `revision`."""
    if not is_clef(checkpoint, revision):
        return None
    if os.path.isdir(checkpoint):
        return checkpoint
    try:
        return str(
            snapshot_download(checkpoint, revision=revision, allow_patterns=list(CLEF_FILES))
        )
    except Exception as e:
        raise FileNotFoundError(
            f"{checkpoint}: a Clef checkpoint (it has {HEAD_CONFIG}), but downloading its files "
            f"failed on the Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


# ------------------------------------------------------------------------------ the engine
@dataclass
class ClefEngineConfig:
    device: str = "cuda"
    model_name: str = "clef-flash"
    # Not used by this engine (no state is cached); kept so /health reports the same keys.
    use_prefix_cache: bool = False
    max_length: int = DEFAULT_MAX_LENGTH


def to_answer(question: Question, option_ids: Sequence[str], probs: Sequence[float]) -> Answer:
    """The answer in the same shape a strands-decider checkpoint returns, from the head's
    per-option probabilities (in prompt order, `option_ids`)."""
    by_id = dict(zip(option_ids, (float(p) for p in probs), strict=True))
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=round(by_id["true"], 4))
    if isinstance(question, ChoiceQuestion):
        names = [str(name) for name in question.criteria]  # the request's order
        p = [by_id[name] for name in names]
        best = max(range(len(p)), key=p.__getitem__)  # the first maximum, in request order
        return ChoiceAnswer(
            choice=names[best],
            probabilities={n: round(x, 4) for n, x in zip(names, p, strict=True)},
            confidence=round(derive_confidence(p), 4),
        )
    levels = [str(i) for i in range(len(question.criteria))]
    p = [by_id[level] for level in levels]
    return ScoreAnswer(
        score=round(sum(i * x for i, x in enumerate(p)), 4),
        legend=dict(zip(levels, question.criteria, strict=True)),
        probabilities={level: round(x, 4) for level, x in zip(levels, p, strict=True)},
        confidence=round(derive_score_confidence(p), 4),
    )


class ClefEngine:
    """Answers System One requests from a Clef checkpoint, all questions of a request in one
    forward. The same surface as `SystemOneEngine`: `evaluate(request)` and `ask(state,
    questions)` return a `SystemOneResponse`. Build one with `load_clef_engine`."""

    def __init__(
        self,
        model: ClefModel,
        processor: Any,
        config: ClefEngineConfig,
        *,
        base_model: str = "Qwen3.5",
    ) -> None:
        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.cfg = config
        self.base_model = base_model
        # A fast tokenizer is not safe to call from two threads at once (see SystemOneEngine).
        self._tok_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def health_fields(self) -> dict[str, Any]:
        """/health's model fields: the base model and the prompt window (no temperature: the
        head's softmax is read as is)."""
        return {
            "base_model": self.base_model,
            "max_length": self.cfg.max_length,
            "temperature": None,
        }

    def encode(self, request: SystemOneRequest) -> EncodedRecord:
        images: list[Any] = []
        if request.images:
            from .h2o_vision import decode_images

            images = decode_images(request.images, MAX_PIXELS)
        with self._tok_lock:
            return encode_record(
                self.tok,
                request.state,
                request.questions,
                images=images,
                processor=self.processor,
                max_length=self.cfg.max_length,
            )

    @torch.inference_mode()
    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        record = self.encode(request)
        logits = self.model(collate(record, self.device))[0]
        answers: dict[str, Answer] = {}
        for q, q_logits in zip(record.questions, logits, strict=True):
            probs = q_logits.float().softmax(-1).tolist()
            answers[q.question_id] = to_answer(
                request.questions[q.question_id], q.option_ids, probs
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=len(record.input_ids), output_tokens=0),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))

    def ask_images(
        self, state: Content, questions: dict[str, Question], images: Sequence[str]
    ) -> SystemOneResponse:
        return self.evaluate(
            SystemOneRequest(state=state, questions=questions, images=list(images))
        )


def _dtype(device: str) -> torch.dtype:
    # bf16 as released; CPU runs it too, but float32 is the safe default there.
    return torch.float32 if device == "cpu" else torch.bfloat16


class ImageTextProcessor:
    """The part of the release's processor (Qwen3VLProcessor) a Clef request uses, without its
    video processor, which needs torchvision: the image processor (pinned to the PIL backend,
    as vision.py and h2o_vision.py are) and the tokenizer. Each `<|image_pad|>` of `text` is
    expanded to the image's token count (its grid's product over merge_size**2) and the image
    tokens are marked 1 in `mm_token_type_ids`, as Qwen3VLProcessor does."""

    image_token = "<|image_pad|>"

    def __init__(self, tokenizer: Any, image_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = int(tokenizer.convert_tokens_to_ids(self.image_token))

    def __call__(
        self, *, text: list[str], images: list[Any], return_tensors: str = "pt"
    ) -> dict[str, Any]:
        batch = self.image_processor(images=images, return_tensors=return_tensors)
        grid = batch["image_grid_thw"]
        merge = int(self.image_processor.merge_size) ** 2
        (prompt,) = text
        pieces = prompt.split(self.image_token)
        if len(pieces) - 1 != len(images):
            raise ValueError(f"{len(pieces) - 1} image placeholders for {len(images)} images")
        expanded = pieces[0]
        for index, piece in enumerate(pieces[1:]):
            expanded += self.image_token * (int(grid[index].prod()) // merge) + piece
        ids = list(self.tokenizer(expanded, add_special_tokens=False).input_ids)
        types = [1 if t == self.image_token_id else 0 for t in ids]
        return {
            "input_ids": torch.tensor([ids]),
            "mm_token_type_ids": torch.tensor([types]),
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": grid,
        }


def load_clef_engine(
    path: str,
    *,
    device: str = "cuda",
    model_name: str = "clef-flash",
    attn_implementation: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> ClefEngine:
    """A `ClefEngine` from the local Clef checkpoint directory `path` (see `resolve`)."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if device == "mlx":
        raise ValueError("a Clef checkpoint runs on torch devices (cuda, mps, cpu), not mlx")
    dtype = _dtype(device)
    extra: dict[str, Any] = {}
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map={"": device}, **extra
    )
    backbone.config.use_cache = False
    head_config = json.loads((Path(path) / HEAD_CONFIG).read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(str(Path(path) / HEAD_WEIGHTS)), strict=True)
    head = head.to(device=device, dtype=dtype)
    from transformers import Qwen2VLImageProcessorPil

    processor = ImageTextProcessor(
        AutoTokenizer.from_pretrained(path), Qwen2VLImageProcessorPil.from_pretrained(path)
    )
    text_config = getattr(backbone.config, "text_config", backbone.config)
    base_model = f"{type(backbone).__name__} ({text_config.num_hidden_layers} layers)"
    return ClefEngine(
        ClefModel(backbone, head).eval(),
        processor,
        ClefEngineConfig(device=device, model_name=model_name, max_length=max_length),
        base_model=base_model,
    )
