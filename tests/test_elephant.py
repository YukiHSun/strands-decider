"""qev checkpoints (baselabs/elephant-1): rendering, block attention, settings, answers, serving.

No released weights are loaded here; the block mask is checked on a tiny random Qwen3.5. The live
check against the release's own runtime is in docs/elephant.md.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from strands_decider import elephant, server
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    derive_confidence,
)


# ------------------------------------------------------------------------------ rendering
def test_noul_renders_yes_first_with_defaults():
    r = elephant.render_question(NoulQuestion(instructions=""))
    assert r.kind == "noul"
    assert r.suffix == (
        f"Question: {elephant.NOUL_DEFAULT_INSTRUCTIONS}\n\nOptions:\n"
        f"A. yes: {elephant.NOUL_DEFAULT_TRUE}\nB. no: {elephant.NOUL_DEFAULT_FALSE}\n\n"
        "Answer with one letter: A, B."
    )
    r = elephant.render_question(
        NoulQuestion(instructions=" Open? ", criteria={"true": "It is open."})
    )
    assert r.suffix.startswith("Question: Open?\n\nOptions:\nA. yes: It is open.\nB. no: the")


def test_choice_and_score_render_as_qev():
    c = elephant.render_question(
        ChoiceQuestion(instructions="Team?", criteria={"billing": "Charges", "fraud": None})
    )
    assert c.suffix == (
        "Question: Team?\n\nOptions:\nA. billing: Charges\nB. fraud\n\nAnswer with one letter: A, B."
    )
    assert c.names == ["billing", "fraud"]
    s = elephant.render_question(ScoreQuestion(instructions="How bad?", criteria=["low", "high"]))
    assert s.suffix == (
        "Rate the state: How bad?\n\nOptions:\nA. level 0: low\nB. level 1: high\n\n"
        "Answer with one letter: A, B."
    )
    assert s.legend == ["low", "high"]


def test_structured_entries_are_indented_json():
    c = elephant.render_question(
        ChoiceQuestion(
            instructions={"ask": "which"}, criteria={"up": {"meaning": "rising"}, "flat": None}
        )
    )
    assert c.suffix.startswith(
        'Question: {\n  "ask": "which"\n}\n\nOptions:\nA. up: {\n  "meaning"'
    )
    assert elephant.render_state({"a": [1]}) == '{\n  "a": [\n    1\n  ]\n}'
    assert elephant.state_block("s") == "<state>\ns\n</state>\n\n"


def test_many_options_use_the_label_line():
    names = {f"o{i}": None for i in range(30)}
    r = elephant.render_question(ChoiceQuestion(instructions="?", criteria=names))
    assert "\nAD. o29" in r.suffix
    assert r.suffix.endswith("AD.")
    assert "Answer with one label: A, B" in r.suffix
    assert elephant.SLOT_LABELS[26:28] == ("AA", "AB")
    assert len(elephant.SLOT_LABELS) == 256


# ------------------------------------------------------------------------------ settings
def _checkpoint(tmp_path, readout, overrides=None, calibration=None):
    (tmp_path / elephant.READOUT_CONFIG).write_text(json.dumps(readout))
    config = {"model_type": "qwen3_5"}
    if overrides is not None:
        config[elephant.OVERRIDES_KEY] = overrides
    (tmp_path / "config.json").write_text(json.dumps(config))
    if calibration is not None:
        (tmp_path / elephant.CALIBRATION).write_text(json.dumps({"temperatures": calibration}))
    return str(tmp_path)


def test_attention_comes_from_the_model_overrides(tmp_path):
    path = _checkpoint(
        tmp_path,
        {"readout": "slot_lm", "head_dim": 256, "n_slots": 256},
        {"attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"}},
    )
    assert elephant.ReadoutConfig.load(path) == elephant.ReadoutConfig("noncausal", "slot_lm")


def test_attention_defaults_to_causal_and_head_readout_is_refused(tmp_path):
    path = _checkpoint(tmp_path, {"readout": "slot_lm"})
    assert elephant.ReadoutConfig.load(path).attention == "causal"
    path = _checkpoint(tmp_path, {"readout": "head"})
    with pytest.raises(ValueError, match="slot_lm"):
        elephant.ReadoutConfig.load(path)


def test_calibration_falls_back_by_kind_then_one(tmp_path):
    path = _checkpoint(tmp_path, {}, calibration={"choice:3": 2.0, "noul": 0.5})
    cal = elephant.Calibration.load(path)
    assert (cal.of("choice", 3), cal.of("choice", 4), cal.of("noul", 2)) == (2.0, 1.0, 0.5)
    assert elephant.Calibration.load(str(tmp_path / "missing")).of("score", 3) == 1.0


def test_detection(tmp_path):
    path = _checkpoint(tmp_path, {})
    assert elephant.resolve(path) == path
    (tmp_path / "strands_decider_config.json").write_text("{}")
    assert not elephant.is_elephant(path)


# ------------------------------------------------------------------------------ block attention
def _tiny_qwen35():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=1,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
    )  # fmt: skip
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    lm = Qwen3_5ForCausalLM(cfg).eval()
    elephant.install_block_mask(lm)
    return lm


def test_block_ids():
    assert elephant.block_ids(2, [4, 3], 5).tolist() == [[0, 0, 1, 1, -1], [0, 0, 1, -1, -1]]


@torch.inference_mode()
def test_blocks_see_themselves_both_ways_and_the_prefix_never_sees_a_question():
    lm = _tiny_qwen35()
    ids = torch.randint(0, 64, (1, 10))
    run = lambda x: lm.model(input_ids=x).last_hidden_state[0]  # noqa: E731
    with elephant.blocks(elephant.block_ids(6, [10], 10)):
        h = run(ids)
        in_prefix, in_question = ids.clone(), ids.clone()
        in_prefix[0, 5] = (in_prefix[0, 5] + 1) % 64
        in_question[0, 8] = (in_question[0, 8] + 1) % 64
        h_prefix, h_question = run(in_prefix), run(in_question)
    assert not torch.allclose(h[0], h_prefix[0])  # token 0 sees token 5 of its block
    assert torch.allclose(h[:6], h_question[:6])  # the prefix does not see the question
    assert not torch.allclose(h[7], h_question[7])  # token 7 sees token 8 of its block
    assert not torch.allclose(h[0], run(ids)[0])  # outside the context: causal again


@torch.inference_mode()
def test_whole_rows_equal_qevs_prefix_cache():
    """qev runs the prefix once and each question from its cache; a whole row under the same
    block mask gives the same hidden state."""
    lm = _tiny_qwen35()
    ids = torch.randint(0, 64, (1, 10))
    with elephant.blocks(elephant.block_ids(6, [10], 10)):
        whole = lm.model(input_ids=ids).last_hidden_state[0, -1]
    with elephant.blocks(elephant.block_ids(6, [6], 6)):
        pre = lm.model(input_ids=ids[:, :6], attention_mask=torch.ones(1, 6, dtype=torch.long))
    with elephant.blocks(elephant.block_ids(6, [10], 10)):
        rest = lm.model(
            input_ids=ids[:, 6:],
            attention_mask=torch.ones(1, 10, dtype=torch.long),
            past_key_values=pre.past_key_values,
        )
    assert torch.allclose(whole, rest.last_hidden_state[0, -1], atol=1e-5)


# ------------------------------------------------------------------------------ the engine
class CharTokenizer:
    """One token per character (ids 1..), letters A..Z and the two-letter labels as one token."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab = {label: i + 1 for i, label in enumerate(elephant.SLOT_LABELS)}
        self.inverse = {i: label for label, i in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        return [1000 + ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inverse.get(i) or chr(i - 1000) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw) -> str:
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<a>"


class StubLM(torch.nn.Module):
    """Hidden state at the last token: a one-hot of the row's length mod 4, so rows differ."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 300, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1, 0] = 3.0  # "A" wins when the hidden state is e0
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.calls = 0
        outer = self

        class Body(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache):
                outer.calls += 1
                n = attention_mask.sum(-1)
                h = torch.zeros(*input_ids.shape, 4)
                h[torch.arange(len(n)), n - 1, 0] = 1.0
                return SimpleNamespace(last_hidden_state=h)

        self.model = Body()

    def get_output_embeddings(self):
        return self.head


def _engine(attention: str = "causal", calibration=None) -> elephant.ElephantEngine:
    return elephant.ElephantEngine(
        StubLM(),
        CharTokenizer(),
        elephant.ReadoutConfig(attention, "slot_lm"),
        elephant.Calibration(calibration or {}),
        elephant.ElephantEngineConfig(device="cpu"),
    )


def test_rows_share_the_prefix_up_to_the_state_block():
    engine = _engine()
    request = SystemOneRequest(
        state="s",
        questions={
            "a": NoulQuestion(instructions="x?"),
            "b": ChoiceQuestion(instructions="y?", criteria={"p": None, "q": None}),
        },
    )
    _, rows, prefix = engine.rows(request)
    text = engine.tok.decode(rows[0][:prefix])
    assert text.endswith("<state>\ns\n</state>\n\n")
    assert engine.tok.decode(rows[1][prefix:]).startswith("Question: y?")


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
    assert engine.lm.calls == 1
    c = response.answers["c"]
    assert c.choice == "a"
    assert c.confidence == pytest.approx(
        derive_confidence(list(c.probabilities.values())), abs=2e-4
    )
    assert response.answers["n"].noul > 0.9  # option A is yes


def test_temperature_flattens():
    req = SystemOneRequest(
        state="s",
        questions={"c": ChoiceQuestion(instructions="?", criteria={"a": None, "b": None})},
    )
    sharp = _engine().evaluate(req).answers["c"].confidence
    flat = _engine(calibration={"choice": 10.0}).evaluate(req).answers["c"].confidence
    assert flat < sharp


def test_server_serves_an_elephant_engine():
    engine = _engine()
    app = server.create_app("baselabs/elephant-1", device="cpu", build_engine=lambda: engine)
    with TestClient(app) as client:
        assert client.get("/health").json()["readout"] == "slot_lm"
        body = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}
        assert client.post("/v1/systemone", json=body).json()["answers"]["q"]["type"] == "noul"


def test_images_are_refused():
    with pytest.raises(ValueError, match="images"):
        _engine().evaluate(
            SystemOneRequest(
                state="s", questions={"q": NoulQuestion(instructions="?")}, images=["x"]
            )
        )
