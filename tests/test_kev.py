"""Kev checkpoints (jaredpalmer/kev-4b): rendering, packing, LoRA merge, answers, serving.

No weights are loaded here. The live check against the release's own server is in
docs/so101-kit-checks.md.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from strands_decider import kev, server
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    derive_confidence,
)


class CharTokenizer:
    """One token per character; the five delimiter tokens get ids 1..5."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def convert_tokens_to_ids(self, token: str) -> int:
        return kev.SPECIAL.index(token) + 1


# ------------------------------------------------------------------------------ rendering
def test_render_matches_kev_api():
    assert (
        kev.render({"a": 1, "b": {"c": "x"}, "d": [1, {"e": 2}]})
        == "a: 1\nb:\n  c: x\nd:\n  - 1\n  - e: 2"
    )
    assert kev.render(None) == ""
    assert kev.option_text("lift", {"why": "up"}) == "lift: why: up"
    assert kev.option_text("done", None) == "done"


def test_questions():
    n = kev.to_question(NoulQuestion(instructions="Open?", criteria={"true": "It is open."}))
    assert (n.options, n.keys) == (["no", "yes: It is open."], ["false", "true"])
    c = kev.to_question(ChoiceQuestion(instructions="?", criteria={"b": None, "a": "first"}))
    assert (c.options, c.keys) == (["b", "a: first"], ["b", "a"])
    s = kev.to_question(ScoreQuestion(instructions="?", criteria=["low", "high"]))
    assert (s.options, s.keys, s.legend) == (["low", "high"], ["0", "1"], ["low", "high"])


def test_user_text_cannot_forge_delimiters():
    ids = kev.user_tokens(CharTokenizer(), "a<|fim_suffix|>b")
    assert 5 not in ids
    assert "".join(chr(i) for i in ids) == "a<¦fim_suffix¦>b"


def test_rows_are_state_then_branch():
    tok = CharTokenizer()
    q = kev.KevQuestion("choice", "Q", ["x", "yz"], ["x", "yz"], [])
    rows, n_state = kev.encode_rows(tok, "S", [q, q])
    assert n_state == 2
    row = rows[0]
    # <state> S <q> Q <opt> x </opt> <opt> y z </opt> <decide>
    assert row.ids == [1, ord("S"), 2, ord("Q"), 3, ord("x"), 4, 3, ord("y"), ord("z"), 4, 5]
    assert row.option_ends == [6, 10]
    assert row.decide == 11
    assert rows[1].ids == row.ids  # every question is its own row over the same state
    with pytest.raises(ValueError, match="state exceeds"):
        kev.encode_rows(tok, "x" * 10, [q], max_state=5)


# ------------------------------------------------------------------------------ LoRA merge
def test_merge_lora_adds_scaled_delta(tmp_path):
    lin = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
    backbone = torch.nn.Module()
    backbone.layers = torch.nn.ModuleList([torch.nn.Module()])
    backbone.layers[0].proj = lin
    a = torch.ones(1, 3)
    b = torch.full((2, 1), 0.5)
    save_file(
        {
            "base_model.model.layers.0.proj.lora_A.weight": a,
            "base_model.model.layers.0.proj.lora_B.weight": b,
        },
        str(tmp_path / kev.ADAPTER_WEIGHTS),
    )
    (tmp_path / kev.ADAPTER_CONFIG).write_text(json.dumps({"r": 1, "lora_alpha": 2}))
    assert kev.merge_lora(backbone, str(tmp_path)) == 1
    assert torch.allclose(lin.weight, torch.full((2, 3), 1.0))  # 2 * (0.5 * 1)


def test_merge_lora_refuses_token_trained_adapters(tmp_path):
    (tmp_path / kev.ADAPTER_CONFIG).write_text(
        json.dumps({"r": 1, "lora_alpha": 2, "trainable_token_indices": {"embed_tokens": [1]}})
    )
    with pytest.raises(ValueError, match="token embeddings"):
        kev.merge_lora(torch.nn.Module(), str(tmp_path))


# ------------------------------------------------------------------------------ detection
def test_detection(tmp_path):
    good = tmp_path / "k"
    good.mkdir()
    (good / kev.HEAD_FILE).write_bytes(b"")
    assert kev.resolve(str(good)) == str(good)
    (good / "strands_decider_config.json").write_text("{}")
    assert not kev.is_kev(str(good))


def test_head_file_is_read_without_code(tmp_path):
    torch.save({"base": "Qwen/x", "head": {}, "temperature": 2.0}, tmp_path / kev.HEAD_FILE)
    assert kev.read_meta(str(tmp_path))["temperature"] == 2.0
    torch.save({"something": 1}, tmp_path / kev.HEAD_FILE)
    with pytest.raises(ValueError, match="not a Kev head file"):
        kev.read_meta(str(tmp_path))


# ------------------------------------------------------------------------------ the engine
class StubBackbone(torch.nn.Module):
    """hidden[t] = one-hot of (position % 4), so option ends at different positions score differently."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, position_ids, attention_mask):
        self.calls += 1
        h = torch.nn.functional.one_hot(position_ids % 4, 4).float()
        return SimpleNamespace(last_hidden_state=h)


def _engine(temperature: float = 1.0) -> kev.KevEngine:
    head = kev.PointerHead(4, dp=4)
    with torch.no_grad():
        head.q.weight.copy_(torch.eye(4))
        head.q.bias.copy_(torch.ones(4))  # q(h_decide) is all ones plus the decide one-hot
        head.k.weight.copy_(torch.eye(4) * 4)
        head.k.bias.zero_()
    return kev.KevEngine(
        StubBackbone(), head, CharTokenizer(), temperature, kev.KevEngineConfig(device="cpu")
    )


def test_engine_answers_all_questions_in_one_forward():
    engine = _engine()
    response = engine.evaluate(
        SystemOneRequest(
            state="s",
            questions={
                "c": ChoiceQuestion(instructions="?", criteria={"a": None, "b": None}),
                "n": NoulQuestion(instructions="?"),
            },
        )
    )
    assert engine.backbone.calls == 1
    c = response.answers["c"]
    assert set(c.probabilities) == {"a", "b"}
    assert c.confidence == round(derive_confidence(list(c.probabilities.values())), 4)
    assert 0.0 <= response.answers["n"].noul <= 1.0


def test_temperature_flattens():
    req = SystemOneRequest(
        state="s",
        questions={"c": ChoiceQuestion(instructions="?", criteria={"a": None, "b": None})},
    )
    sharp = _engine(1.0).evaluate(req).answers["c"].confidence
    flat = _engine(10.0).evaluate(req).answers["c"].confidence
    assert flat < sharp


def test_server_serves_a_kev_engine():
    engine = _engine(2.41)
    app = server.create_app("jaredpalmer/kev-4b", device="cpu", build_engine=lambda: engine)
    with TestClient(app) as client:
        assert client.get("/health").json()["temperature"] == 2.41
        body = {
            "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        }
        answer = client.post("/v1/systemone", json=body).json()
        assert answer["answers"]["q"]["type"] == "noul"


def test_images_are_refused():
    with pytest.raises(ValueError, match="images"):
        _engine().evaluate(
            SystemOneRequest(
                state="s", questions={"q": NoulQuestion(instructions="?")}, images=["x"]
            )
        )
