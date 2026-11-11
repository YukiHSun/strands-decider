# Kev (jaredpalmer/kev-4b)

This fork branch also reads [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b), a 4B
decision model (Apache-2.0): a LoRA adapter and a pointer head on `Qwen/Qwen3.5-4B-Base`. It is read
from Python (`load_engine`), from `strands-decider ask` and from `strands-decider serve`, through the
same `POST /v1/systemone` API, with the same answer shape as a strands-decider checkpoint's. No code
from the release is run: `src/strands_decider/kev.py` carries its rendering, packing and readout,
ported from the release's own code ([jaredpalmer/kev](https://github.com/jaredpalmer/kev),
Apache-2.0).

A Kev checkpoint is recognised by `head.pt` and no `strands_decider_config.json`. `head.pt` is read
with `torch.load(..., weights_only=True)`, so no code is unpickled.

## Serving

```bash
strands-decider serve jaredpalmer/kev-4b --device mps        # or cuda / cpu
strands-decider ask jaredpalmer/kev-4b --device mps --state "The door is open." --noul "Is the door open?"
```

- The adapter's base (`base` at `base_revision` in `head.pt`: `Qwen/Qwen3.5-4B-Base` at
  `1001bb4d`, about 9 GB) is downloaded from the Hub, and the LoRA is merged into it at load time
  (delta in fp32, one rounding to bf16), as the release does by default.
- Torch devices only (`cuda`, `mps`, `cpu`); `--device mlx`, `--device-map` and `--vision` are
  refused. The release's server runs it through MLX on Apple silicon; this branch uses MPS.
- Supported: Kev checkpoints on hybrid (Qwen3.5) bases, LoRA or full weights. Refused: option
  isolation, adapters with trained token embeddings, attention-only bases (they need the packed
  block-causal form).
- `--noul-floor` does not apply; the head's calibration temperature (2.41 for kev-4b) is applied to
  every answer, as in the release.

## How a request is read

As the release's server does (`kev.api`, `kev.model`):

- the state, instructions and options are rendered to text (`render`: field names kept as labels,
  lists as `- ` items); a choice option is `name` or `name: <description>`, a yes/no question the
  options `no[: <false>]` and `yes[: <true>]`, a score question its levels as given;
- caller text can never produce the delimiter tokens (`<|name|>` is rewritten to `<¦name¦>`);
- every question is one causal row: `<|fim_prefix|>` state, then `<|fim_middle|>` instruction,
  `<|box_start|>` option `<|box_end|>` for each option, and `<|fim_suffix|>` (the decide token);
- logits = q(h_decide) · k(h_option_end) / sqrt(256), divided by the temperature and softmaxed.
  All rows of a request run in one forward.

`confidence` is `derive_confidence` for choice (the release uses the same formula) and
`derive_score_confidence` for score (the release uses another score formula).

## How it was checked

Revision `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` on `Qwen/Qwen3.5-4B-Base` `1001bb4d`,
`--device mps` on an Apple M-series Mac (36 GB), 2026-11-11.

**Against the release's own server** (`KEV_BACKEND=torch`, so both run PyTorch on MPS in bf16):
30 requests (the 26 that the SO-101 kit sent, a score question, a 12-option choice, a yes/no
question without instructions and a JSON state) gave the same choice and the same yes/no side on
all 46 answers; the score question's expected level was 1.137 here and 1.150 there. The largest
difference in any probability was 0.026. The release runs the state once and continues each
question from its cache, and pads rows to multiples of 64 on MPS; this branch runs every row whole,
so the two agree to bf16 rounding, not bit for bit.

**On the SO-101 kit** ([docs/so101-kit-checks.md](so101-kit-checks.md)): the same skill as the
release's server on 19 of 19 recorded requests, both against its MLX backend (the backend it picks
on a Mac) and its torch backend. The pass/stop verdict matched the MLX backend on 19 of 19 and the
torch backend on 18 of 19: one confidence sat on the 0.45 gate (0.444 here, 0.44 on MLX, 0.457 on
torch). 7 of 15 successful steps passed, and all 7 deliberately failed observations stopped.
Median time per decision: 1.2 s here, 2.7 s on the release's torch backend, 0.8 s on its MLX one.
