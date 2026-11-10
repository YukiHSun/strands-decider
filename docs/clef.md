# Clef (Cloudflare/clef-flash)

This fork branch also reads [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash),
a 9B decision model from Cloudflare built on Qwen3.5-9B (Apache-2.0), from Python (`load_engine`),
from `strands-decider ask` and from `strands-decider serve`. It is served through the same
`POST /v1/systemone` API, and its answers have the same shape as a strands-decider checkpoint's.
No code from the model repo is run (`trust_remote_code` is not used): `src/strands_decider/clef.py`
carries the release's prompt layout and joint schema head, ported from its reference
implementation (`joint_schema_model.py`, Apache-2.0).

A Clef checkpoint is recognised by `joint_head_config.json` (with `joint_head.safetensors`) and no
`strands_decider_config.json`.

## Serving

```bash
strands-decider serve Cloudflare/clef-flash --device cuda            # a GPU (bf16, about 19 GB)
strands-decider serve Cloudflare/clef-flash --device mps             # Apple silicon, 32 GB or more
strands-decider serve /path/to/clef-flash --device mps --model-name clef-flash
```

- Torch devices only (`cuda`, `mps`, `cpu`). `--device mlx`, `--device-map` and
  `--engine llama-server` are not supported for it; `--attention` and `--noul-floor` do not
  apply (there is no yes/no floor).
- Images are always accepted (`images` in the request, base64 or `data:` URIs): the release has its
  vision tower merged in. Each image is shrunk to at most 1024 x 1024 pixels and preprocessed with
  the PIL image processor, so torchvision is not needed (the release's own `AutoProcessor` needs
  it for its video processor).
- All questions of a request are answered in one forward of the backbone (the head reads them
  jointly). The prompt window is 16 384 tokens; as in the reference, the state is truncated to
  what the schema leaves of it.

## Answers

The same fields as a strands-decider checkpoint's:

| Type | Field | From the head's per-option softmax |
|---|---|---|
| noul | `noul` | P(true), with no floor |
| choice | `choice`, `probabilities` | as the reference; options in the request's order |
| choice | `confidence` | `derive_confidence` (uniform 0, one-hot 1), as for the other engines of this package. The reference returns the raw max-probability instead |
| score | `score`, `probabilities`, `legend` | as the reference |
| score | `confidence` | `derive_score_confidence` |

## How it was checked

Revision `fb4417d31c1d0f9e7e77f1141b40aaede7491e00` of `Cloudflare/clef-flash`, `--device mps`
(Apple M-series, 36 GB), 2026-11-10.

**Against the reference implementation.** 27 requests (the 26 that the SO-101 kit below sent, and
one with an image) were answered by this engine and by the release's `systemone()` on the same
loaded weights: every answer matched, and the largest difference in any probability was 0.0.

**On a robot-arm decision task.** The SO-101 pick-and-place kit
[YukiHSun/pai-final-so101-open-decider](https://github.com/YukiHSun/pai-final-so101-open-decider)
asks a decision model for the next skill and stops on a confidence gate (0.45), a safety question
(yes at 0.60 or more) and a transition check. Its comparison scripts were run unchanged against
`strands-decider serve ~/models/clef-flash --device mps`, with the hosted Jev 1.14 as the
reference:

| | Clef-Flash (this branch) | Jev 1.14 (hosted) |
|---|---|---|
| Same skill as hosted Jev, 19 recorded requests (the kit's wording) | 16 / 19 | — |
| Successful steps that pass every gate (15, the kit's wording) | 13 / 15 | 15 / 15 |
| Same skill as the recorded run, revised wording (`strict_anchored`) | 16 / 19 | 18 / 19 |
| Successful steps that pass every gate (15, `strict_anchored`) | 12 / 15 | 15 / 15 |
| Deliberately failed observations stopped (7) | 7 / 7 | 7 / 7 |
| Median time per decision | 1.8 s (MPS) | 0.16 s |

Where Clef-Flash differed: right after the lift of the pick-and-place task it chose `lower`
(confidence 0.35, so the gate stopped it) instead of `secure_grasp`; after `retreat` it chose
`retreat` again instead of `done`; and with the revised wording, after the lift of the cube-lift
task it chose `secure_grasp`, a skill of the other task offered in the same list. The 1.8 s is
this Mac's MPS; Cloudflare reports a 38.8 ms median on its own serving on H200-class GPUs.
