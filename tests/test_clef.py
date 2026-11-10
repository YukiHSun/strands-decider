"""Clef checkpoints (Cloudflare/clef-flash): prompt layout, answers, detection and serving.

No weights are loaded here: the tokenizer is one token per character and the backbone is a stub.
The live check against the release's reference implementation is in docs/clef.md.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from strands_decider import clef, server
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    derive_confidence,
    derive_score_confidence,
)


class CharTokenizer:
    """One token per character (its code point), so spans can be read back as text."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def _text(record: clef.EncodedRecord, span: tuple[int, int]) -> str:
    return "".join(chr(t) for t in record.input_ids[span[0] : span[1]])


# ------------------------------------------------------------------------------ prompt layout
def test_prompt_follows_the_reference_layout():
    record = clef.encode_record(
        CharTokenizer(),
        {"b": 1, "a": "x"},
        {"q": NoulQuestion(instructions="Is it lifted?")},
    )
    text = "".join(chr(t) for t in record.input_ids)
    assert text.startswith(
        "<|im_start|>system\n" + clef.SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\nSTATE:\n"
        '{"a":"x","b":1}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: q\nTYPE: noul\nINSTRUCTION: '
    )
    assert text.endswith(
        "END FIELD\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "JOINT SCHEMA DECISIONS:"
    )


def test_spans_cover_the_instruction_and_each_option():
    record = clef.encode_record(
        CharTokenizer(),
        "state",
        {
            "skill": ChoiceQuestion(
                instructions="Pick the next skill.",
                criteria={"lift": "Lift the cube.", "approach": None},
            ),
            "lifted": NoulQuestion(instructions="Lifted?", criteria={"true": "Off the table."}),
            "grip": ScoreQuestion(instructions="How firm?", criteria=["loose", "firm"]),
        },
    )
    skill, lifted, grip = record.questions
    assert _text(record, skill.question_span) == "Pick the next skill."
    # Choice options are listed sorted by id, as the reference lists them.
    assert skill.option_ids == ("approach", "lift")
    assert [_text(record, s) for s in skill.option_spans] == [
        '{"option_id":"approach"}',
        '{"description":"Lift the cube.","option_id":"lift"}',
    ]
    # Noul: true then false; a given description replaces the default, the other stays.
    assert lifted.option_ids == ("true", "false")
    assert [_text(record, s) for s in lifted.option_spans] == [
        '{"description":"Off the table.","option_id":"true"}',
        '{"description":"' + clef._NOUL_DEFAULTS["false"] + '","option_id":"false"}',
    ]
    assert grip.option_ids == ("0", "1")
    assert [q.question_type for q in record.questions] == [1, 0, 2]


def test_a_null_noul_description_keeps_the_default():
    q = NoulQuestion(instructions="?", criteria={"true": None})
    assert clef.question_options(q)[0] == ("true", clef._NOUL_DEFAULTS["true"])


def test_empty_instructions_fall_back_to_the_question_id():
    record = clef.encode_record(CharTokenizer(), "", {"is_open": NoulQuestion(instructions="")})
    assert _text(record, record.questions[0].question_span) == "is_open"


def test_state_is_truncated_to_the_window_and_a_long_schema_is_refused():
    questions = {"q": NoulQuestion(instructions="?")}
    short = clef.encode_record(CharTokenizer(), "", questions)
    fixed = len(short.input_ids)
    record = clef.encode_record(CharTokenizer(), "s" * 50, questions, max_length=fixed + 10)
    assert len(record.input_ids) == fixed + 10
    with pytest.raises(ValueError, match="before state"):
        clef.encode_record(CharTokenizer(), "", questions, max_length=fixed - 1)


# ------------------------------------------------------------------------------ answers
def test_choice_answer_is_in_request_order_with_derived_confidence():
    q = ChoiceQuestion(instructions="?", criteria={"lift": None, "approach": None, "grasp": None})
    # Prompt order (sorted): approach, grasp, lift.
    answer = clef.to_answer(q, ("approach", "grasp", "lift"), [0.2, 0.1, 0.7])
    assert answer.choice == "lift"
    assert list(answer.probabilities) == ["lift", "approach", "grasp"]
    assert answer.probabilities == {"lift": 0.7, "approach": 0.2, "grasp": 0.1}
    assert answer.confidence == round(derive_confidence([0.7, 0.2, 0.1]), 4)


def test_noul_answer_is_p_true_without_a_floor():
    answer = clef.to_answer(NoulQuestion(instructions="?"), ("true", "false"), [0.55, 0.45])
    assert answer.noul == 0.55


def test_score_answer_is_the_expected_level():
    q = ScoreQuestion(instructions="?", criteria=["low", "mid", "high"])
    answer = clef.to_answer(q, ("0", "1", "2"), [0.1, 0.3, 0.6])
    assert answer.score == round(0.3 + 1.2, 4)
    assert answer.legend == {"0": "low", "1": "mid", "2": "high"}
    assert answer.confidence == round(derive_score_confidence([0.1, 0.3, 0.6]), 4)


# ------------------------------------------------------------------------------ the head
def test_head_returns_one_logit_per_option():
    torch.manual_seed(0)
    head = clef.JointSchemaHead(
        hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16
    ).eval()
    record = clef.encode_record(
        CharTokenizer(),
        "s",
        {
            "a": ChoiceQuestion(instructions="?", criteria={"x": None, "y": None, "z": None}),
            "b": NoulQuestion(instructions="?"),
        },
    )
    batch = clef.collate(record, torch.device("cpu"))
    hidden = torch.randn(1, len(record.input_ids), 16)
    embeddings = torch.randn(200, 16)
    with torch.inference_mode():
        logits = head(hidden, batch["input_ids"], batch["attention_mask"], [record], embeddings)
    assert [t.shape[0] for t in logits[0]] == [3, 2]


# ------------------------------------------------------------------------------ detection
def _write_clef_dir(path, *, decider_config: bool = False):
    path.mkdir()
    (path / clef.HEAD_CONFIG).write_text(json.dumps({"hidden_size": 16}))
    (path / clef.HEAD_WEIGHTS).write_bytes(b"")
    if decider_config:
        (path / "strands_decider_config.json").write_text("{}")
    return str(path)


def test_a_local_clef_checkpoint_is_detected(tmp_path):
    path = _write_clef_dir(tmp_path / "clef")
    assert clef.is_clef(path)
    assert clef.resolve(path) == path


def test_a_decider_checkpoint_is_not_clef(tmp_path):
    assert not clef.is_clef(_write_clef_dir(tmp_path / "decider", decider_config=True))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert clef.resolve(str(empty)) is None


def test_a_hub_id_that_cannot_be_fetched_is_not_clef(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(clef, "hf_hub_download", fail)
    assert not clef.is_clef("org/not-there")


# ------------------------------------------------------------------------------ serving
class StubBackbone(torch.nn.Module):
    """Stands in for ClefModel: logits that favour the last option of every question."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        record = batch["records"][0]
        return [[torch.arange(len(q.option_ids), dtype=torch.float32) for q in record.questions]]


def _engine() -> clef.ClefEngine:
    processor = SimpleNamespace(tokenizer=CharTokenizer())
    return clef.ClefEngine(
        StubBackbone(),  # type: ignore[arg-type]
        processor,
        clef.ClefEngineConfig(device="cpu", model_name="clef-flash"),
        base_model="stub",
    )


def test_engine_answers_every_question_in_one_forward():
    engine = _engine()
    response = engine.evaluate(
        SystemOneRequest(
            state="s",
            questions={
                "skill": ChoiceQuestion(instructions="?", criteria={"b": None, "a": None}),
                "ok": NoulQuestion(instructions="?"),
            },
        )
    )
    assert engine.model.calls == 1
    # Prompt order is a, b: the stub favours b.
    assert response.answers["skill"].choice == "b"
    # Noul's prompt order is true, false: the stub favours false.
    assert response.answers["ok"].noul < 0.5
    assert response.usage.output_tokens == 0


def test_server_serves_a_clef_engine_with_the_decider_health_keys():
    engine = _engine()
    app = server.create_app("Cloudflare/clef-flash", device="cpu", build_engine=lambda: engine)
    with TestClient(app) as client:
        # Warm-up: the three kinds, then all three in one request (no per-count shapes).
        assert engine.model.calls == 4
        health = client.get("/health").json()
        assert health["base_model"] == "stub"
        assert health["max_length"] == clef.DEFAULT_MAX_LENGTH
        assert health["temperature"] is None
        body = {
            "state": "s",
            "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None}}
            },
        }
        answer = client.post("/v1/systemone", json=body).json()
        assert answer["model"] == "clef-flash"
        assert answer["answers"]["q"]["choice"] == "y"


def test_mlx_is_refused_for_a_clef_checkpoint():
    with pytest.raises(ValueError, match="not mlx"):
        clef.load_clef_engine("unused", device="mlx")


def test_image_text_processor_expands_each_image_to_its_token_count():
    class Tok(CharTokenizer):
        def __call__(self, text, add_special_tokens=False):
            # "<|image_pad|>" is one token (id 7); every other character is its own.
            ids = []
            for i, part in enumerate(text.split("<|image_pad|>")):
                if i:
                    ids.append(7)
                ids.extend(ord(c) for c in part)
            return SimpleNamespace(input_ids=ids)

        def convert_tokens_to_ids(self, token):
            return 7

    class ImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors="pt"):
            grid = torch.tensor([[1, 4, 4], [1, 2, 4]])[: len(images)]
            return {"pixel_values": torch.zeros(3, 1), "image_grid_thw": grid}

    proc = clef.ImageTextProcessor(Tok(), ImageProcessor())
    out = proc(text=[clef.IMAGE_PLACEHOLDER * 2 + "\n"], images=[object(), object()])
    ids = out["input_ids"][0].tolist()
    assert ids.count(7) == 4 + 2  # 16 / 4 and 8 / 4
    assert out["mm_token_type_ids"][0].tolist() == [1 if t == 7 else 0 for t in ids]
    with pytest.raises(ValueError, match="placeholders"):
        proc(text=[clef.IMAGE_PLACEHOLDER], images=[object(), object()])
