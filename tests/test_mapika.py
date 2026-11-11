"""Mapika decider checkpoints (Mapika/decider-4b): rendering, rows, temperatures, answers, serving.

No weights are loaded here: the tokenizer is one token per character and the backbone is a stub.
The live check against the release's own server is in docs/so101-kit-checks.md.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from strands_decider import mapika, server
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    derive_confidence,
)


class CharTokenizer:
    """One token per character (its code point); two-letter labels are not single tokens."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class LabelTokenizer(CharTokenizer):
    """As CharTokenizer, but every one- or two-letter upper-case string is one token, so the
    full table of 255 labels exists."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text.isalpha() and text.isupper() and len(text) <= 2:
            return [1000 + sum(ord(c) * 100**i for i, c in enumerate(text))]
        return super().encode(text)


def _text(ids: list[int]) -> str:
    return "".join(chr(t) for t in ids)


# ------------------------------------------------------------------------------ rendering
def test_choice_options_name_and_description():
    r = mapika.render_question(
        ChoiceQuestion(instructions="Next skill?", criteria={"lift": "Raise it.", "done": None})
    )
    assert (r.kind, r.text, r.options, r.names) == (
        "choice",
        "Next skill?",
        ["lift: Raise it.", "done"],
        ["lift", "done"],
    )


def test_noul_is_no_then_yes_with_descriptions():
    r = mapika.render_question(
        NoulQuestion(instructions="Lifted?", criteria={"true": "Off the table."})
    )
    assert r.options == ["no", "yes: Off the table."]
    q = NoulQuestion(instructions="", criteria={"false": "Still down."})
    assert mapika.render_question(q).text == mapika.NOUL_WITHOUT_INSTRUCTIONS
    with pytest.raises(ValueError, match="criteria must describe"):
        mapika.render_question(NoulQuestion(instructions=""))


def test_state_json_annotates_long_arrays():
    assert mapika.render_state("s") == "s"
    rendered = json.loads(mapika.render_state({"a": list(range(8)), "b": [1, 2]}))
    assert rendered["a"][3] == {"_index": 3, "value": 3}
    assert rendered["b"] == [1, 2]


def test_score_becomes_one_yes_no_row_per_level():
    r = mapika.render_question(
        ScoreQuestion(instructions="How firm?", criteria=["loose", "2: firm"])
    )
    rows, index = mapika.plan_rows(
        {"grip": r, "ok": mapika.render_question(NoulQuestion(instructions="?"))}
    )
    assert index == [("grip", "iso", 0, 2), ("ok", "list", 2, 1)]
    assert rows[1] == (
        "How firm?\nProposed answer: firm\nDoes the proposed answer fit?",
        ["no", "yes"],
        "score",
    )
    assert rows[2][2] == "noul"


def test_question_piece_narrow_layout():
    tok = LabelTokenizer()
    piece = mapika.question_piece(tok, mapika.Labels(tok), "Q?", ["x", "y: z"])
    assert _text(piece) == "\n\nQuestion: Q?\nOptions:\n(A) x\n(B) y: z\nAnswer: ("


def test_question_piece_wide_layout_uses_one_label_token_per_option():
    tok = LabelTokenizer()
    labels = mapika.Labels(tok)
    options = [f"o{i}" for i in range(12)]
    piece = mapika.question_piece(tok, labels, "Q?", options)
    for j in range(12):
        assert labels.ids[j] in piece
    assert _text(piece[-10:]) == "\nAnswer: ("


def test_labels_need_single_tokens():
    with pytest.raises(ValueError, match="single-token label"):
        mapika.Labels(CharTokenizer())


# ------------------------------------------------------------------------------ temperatures
def test_temperature_by_type_and_by_options():
    t = mapika.Temperatures.from_config(
        {"temperature": 2.0, "temperature_by_type": {"noul": 0.05, "choice": 1.5}}
    )
    assert (t.of("noul", 2), t.of("choice", 3), t.of("score", 2)) == (0.05, 1.5, 2.0)
    o = mapika.Temperatures.from_config({"temperature_by_options": {"a": 1.0, "b": -1.0}})
    assert o.of("choice", 100) == 0.05
    with pytest.raises(ValueError, match="cannot both"):
        mapika.Temperatures.from_config(
            {"temperature_by_type": {"noul": 1}, "temperature_by_options": {"a": 1, "b": 1}}
        )
    with pytest.raises(ValueError, match="unknown answer type"):
        mapika.Temperatures.from_config({"temperature_by_type": {"bool": 1}})


def test_chat_layout_is_refused():
    with pytest.raises(ValueError, match="only the plain layout"):
        mapika.check_layout({"layout": "chat"})
    mapika.check_layout({"layout": "plain"})


# ------------------------------------------------------------------------------ answers
def test_answers_keep_the_package_shape():
    r = mapika.render_question(
        ChoiceQuestion(instructions="?", criteria={"a": None, "b": None, "c": None})
    )
    a = mapika.to_answer(r, [0.2, 0.7, 0.1])
    assert a.choice == "b"
    assert a.confidence == round(derive_confidence([0.2, 0.7, 0.1]), 4)
    n = mapika.to_answer(mapika.render_question(NoulQuestion(instructions="?")), [0.3, 0.7])
    assert n.noul == 0.7


# ------------------------------------------------------------------------------ detection
def test_detection(tmp_path):
    good = tmp_path / "m"
    good.mkdir()
    (good / mapika.DECIDER_CONFIG).write_text("{}")
    assert mapika.resolve(str(good)) == str(good)
    (good / "strands_decider_config.json").write_text("{}")
    assert not mapika.is_mapika(str(good))


def test_a_hub_id_that_cannot_be_fetched_is_not_mapika(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(mapika, "hf_hub_download", fail)
    assert not mapika.is_mapika("org/not-there")


# ------------------------------------------------------------------------------ the engine
class StubLM(torch.nn.Module):
    """hidden state = one-hot of the last token's code point modulo 4, so each row's logits over
    the label rows are known; output embedding rows are one-hot too."""

    def __init__(self, label_ids: list[int]) -> None:
        super().__init__()
        self.calls = 0
        vocab = max(label_ids) + 1
        self.emb = torch.nn.Embedding(vocab, 4)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.emb.weight[label_ids[0], 0] = 5.0  # label A
            self.emb.weight[label_ids[1], 1] = 5.0  # label B
        self.model = self

    def get_output_embeddings(self):
        return self.emb

    def forward(self, input_ids, use_cache=False):
        self.calls += 1
        hidden = torch.zeros(*input_ids.shape, 4)
        hidden[..., 1] = 1.0  # every slot favours label B
        return SimpleNamespace(last_hidden_state=hidden)


def _engine(cfg=None) -> mapika.MapikaEngine:
    tok = LabelTokenizer()
    labels = mapika.Labels(tok)
    return mapika.MapikaEngine(
        StubLM(labels.ids),
        tok,
        cfg or {"temperature": 1.0, "version": "stub"},
        mapika.MapikaEngineConfig(device="cpu", model_name="decider-4b"),
    )


def test_engine_answers_every_question_in_one_forward():
    engine = _engine()
    response = engine.evaluate(
        SystemOneRequest(
            state={"x": 1},
            questions={
                "skill": ChoiceQuestion(instructions="?", criteria={"a": None, "b": None}),
                "ok": NoulQuestion(instructions="?"),
                "grip": ScoreQuestion(instructions="?", criteria=["low", "high"]),
            },
        )
    )
    assert engine.lm.calls == 1
    assert response.answers["skill"].choice == "b"  # label B
    assert response.answers["ok"].noul > 0.99  # option B of a noul is "yes"
    # Both score levels fit equally (each row favours "yes"), so the score is the middle.
    assert response.answers["grip"].score == 0.5


def test_images_are_refused():
    with pytest.raises(ValueError, match="images"):
        _engine().evaluate(
            SystemOneRequest(
                state="s", questions={"q": NoulQuestion(instructions="?")}, images=["x"]
            )
        )


def test_server_serves_a_mapika_engine():
    engine = _engine({"temperature": 2.0, "temperature_by_type": {"noul": 0.05}, "version": "v"})
    app = server.create_app("Mapika/decider-4b", device="cpu", build_engine=lambda: engine)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["base_model"] == "v"
        assert health["temperature"] == {"choice": 2.0, "noul": 0.05, "score": 2.0}
        body = {
            "state": "s",
            "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None}}
            },
        }
        answer = client.post("/v1/systemone", json=body).json()
        assert answer["model"] == "decider-4b"
        assert answer["answers"]["q"]["choice"] == "y"
