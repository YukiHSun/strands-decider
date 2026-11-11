# Elephant (baselabs/elephant-1)

This fork branch also reads [`baselabs/elephant-1`](https://huggingface.co/baselabs/elephant-1), a 9B
decision model by Base Labs at Baseten (MIT): a full fine-tune of Qwen3.5-9B. It is read from
Python (`load_engine`), from `strands-decider ask` and from `strands-decider serve`, through the
same `POST /v1/systemone` API, with the same answer shape as a strands-decider checkpoint's. No
code from the release is run: `src/strands_decider/elephant.py` carries the prompt rendering, the
block attention and the readout of the runtime the release names (OneJev `qev` at `81ce62f` with
the release's `serve.patch`, [OmniJev/OneJev](https://github.com/OmniJev/OneJev), Apache-2.0).

A qev checkpoint is recognised by `readout.json` and no `strands_decider_config.json`. Only its
weights, configs and tokenizer are downloaded; the release's `engine.py`, `setup.sh` and
`serve.patch` are not.

## Serving

```bash
strands-decider serve baselabs/elephant-1 --device mps --port 8748     # or --device cuda / cpu
```

The first start downloads about 19 GB (bf16). Then:

```bash
curl -s http://127.0.0.1:8748/health
# {"status":"ok","model":"elephant-1",...,"attention":"noncausal","readout":"slot_lm",...}

curl -s http://127.0.0.1:8748/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}}
  }
}'
```

One question from the command line, with the same loading rules:

```bash
strands-decider ask baselabs/elephant-1 --device mps --state "The door is open." --noul "Is the door open?"
```

- Memory: about 19 GB of weights in bf16, plus the answer letters' rows of the LM head in fp32.
  On a Mac, `--device mps` needs a 32 GB+ machine; on a GPU, 24 GB is enough (`--device cuda`).
- Torch devices only (`cuda`, `mps`, `cpu`); `--device mlx`, `--device-map` and `--vision` are
  refused. The release reads images; this branch serves it as text only.
- The attention must be a mask-based implementation (sdpa, the default, or eager); the block mask
  is not available with flash attention.
- `--noul-floor` does not apply. Without `calibration.json` (elephant-1 ships none) every answer is
  read at temperature 1.0, as in the release.

## How a request is read

As qev's `DecisionEngine` does by default:

- every question is one chat in the checkpoint's template (generation prompt, thinking off): the
  system turn `Apply the question to the state. Choose exactly one of the listed options. Respond
  with only its uppercase letter, with no explanation or reasoning.`, then a user turn of
  `<state>\n<state>\n</state>\n\n` and the question:
  `Question: <text>\n\nOptions:\nA. <option>\n...\n\nAnswer with one letter: A, B.`;
- a yes/no question has the options `A. yes: <true>` and `B. no: <false>` (qev's defaults when the
  criteria are missing); a choice question `<letter>. <name>[: <description>]`; a score question
  (`Rate the state: ...`) `<letter>. level <i>: <level>`; above 26 options the labels go on to
  two letters (`AA`, `AB`, ...), each one token;
- the state and the questions are two blocks. In the full-attention layers a token sees every
  token of its own block and of the blocks before it, and none after
  (`config.json`: `modeling_overrides.attention.full_attention = block_bidirectional`); the
  Gated DeltaNet layers stay causal. The state block ends where the questions' prompts stop
  sharing tokens, at or before `</state>\n\n`;
- the answer is the LM head's logits (fp32) for the answer letters at the prompt's last token,
  softmaxed.

qev runs the state block once and each question from its forked cache; this branch runs every
question's prompt whole under the same mask, in one padded forward for all questions of a
request. The two compute the same thing (on a small random Qwen3.5 they agree to 2e-7, see
`tests/test_elephant.py`). `confidence` is `derive_confidence` for choice (the same formula as
qev's) and `derive_score_confidence` for score (qev uses another score formula).

## How it was checked

Revision `b3f8a93462fb1051a9de031e94a7ff380e254072`, 2026-11-11.

**The reference.** The release's runtime needs a CUDA GPU, so it was run on a Colab L4 (24 GB)
exactly as the release's `setup.sh` installs it: OneJev `81ce62f` + `serve.patch`,
transformers 5.19.0, flash-linear-attention 0.5.2, causal-conv1d 1.7.0, torch 2.14.1+cu130. It
read the checkpoint with `noncausal` attention and the `slot_lm` readout. One change was needed on
the 24 GB card: qev's float32 copy of the LM head was made through the CPU instead of on the GPU
(the same bf16-to-fp32 cast; qev's own copy runs out of memory there).

**Rendering.** For the 67 questions of the 42 requests below, the chat messages this branch builds
are identical, character for character, to those of qev's `prompt.py`.

**Answers.** The same 42 requests (the 37 the SO-101 kit sends, a score question, a 12-option
choice, a yes/no question without instructions and a JSON state) went to the reference on the L4
and to `strands-decider serve ~/models/elephant-1 --device mps` on an Apple M-series Mac (36 GB):
all 67 answers chose the same option and the same yes/no side; the largest difference in any
probability was 0.026 (bf16 on different kernels and hardware).

**On the SO-101 kit** ([docs/so101-kit-checks.md](so101-kit-checks.md)): the same skill and the same
pass/stop verdict as the reference on 19 of 19 recorded steps and on 7 of 7 failed observations
(confidence within 0.027), and on 19 of 19 steps with the revised wording. 15 of 15 successful
steps passed, no wrong skill passed, and all 7 failures stopped, the same as hosted Jev 1.14.
Median time per decision: 2.2 s here (MPS), 0.42 s for the reference on the L4.
