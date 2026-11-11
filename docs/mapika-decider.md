# Mapika decider (Mapika/decider-4b)

This fork branch also reads [`Mapika/decider-4b`](https://huggingface.co/Mapika/decider-4b), a 4B
decision model built on Qwen3.5-4B-Base (weights Apache-2.0), from Python (`load_engine`), from
`strands-decider ask` and from `strands-decider serve`. It is served through the same
`POST /v1/systemone` API, and its answers have the same shape as a strands-decider checkpoint's.
No code from the release is run: `src/strands_decider/mapika.py` carries its prompt rendering and
readout, ported from the release's own server ([Mapika/decider](https://github.com/Mapika/decider),
Apache-2.0).

A Mapika decider checkpoint is recognised by `decider_config.json` and no
`strands_decider_config.json`.

## Serving

```bash
strands-decider serve Mapika/decider-4b --device mps        # or cuda / cpu; bf16, about 8.5 GB
strands-decider ask Mapika/decider-4b --device mps --state "The door is open." --noul "Is the door open?"
```

- Torch devices only (`cuda`, `mps`, `cpu`); `--device mlx`, `--device-map` and `--vision` are
  refused. The release's `vision/` folder is not downloaded: images are not supported here (use
  the release's server with `DECIDER_IMAGES=1` for images).
- Only the plain prompt layout is read (every released decider-4b uses it); a checkpoint whose
  `decider_config.json` names the chat layout is refused.
- `--noul-floor` does not apply. decider-4b's own `decider_config.json` reads yes/no answers at
  temperature 0.05, so they come out near 0 or 1; `/health` reports the temperature of each answer
  type.

## How a request is read

As the release's server does by default:

- every question is its own row: `Context:\n<state>`, then `\n\nQuestion: <text>\nOptions:`, the
  options as `\n(A) ...` lines, and `\nAnswer: (`; above 10 options every label is one token
  (A..Z, then two-letter labels);
- a JSON state is serialised compactly, with each element's position written into arrays of 8 or
  more items;
- a choice option renders as `name` or `name: <description>`; a yes/no question as the options
  `no[: <false>]` and `yes[: <true>]`;
- a score question becomes one yes/no row per level ("Proposed answer: <level>"), and the levels'
  P(yes) are normalised into the score distribution;
- the answer is the language model's own logits for the label tokens at the final `(`, divided by
  the answer type's temperature and softmaxed. All rows of a request run in one forward.

The answer shape follows this package: `confidence` is `derive_confidence` for choice (the release
uses the same formula) and `derive_score_confidence` for score (the release uses another formula
and rounds the score to 2 places).

## How it was checked

Revision `585473dec4fa93f0325c2bf02dda7273b61f44c1` (the release server reports
`decider-4b-v2.1`), `--device mps` on an Apple M-series Mac (36 GB), 2026-11-11.

**Against the release's own server.** The same 30 requests (the 26 that the SO-101 kit below sent,
and 4 more for a score question, a 12-option choice, a yes/no question without instructions and a
JSON state with an annotated array) went to the release's server and to
`strands-decider serve ~/models/decider-4b --device mps`. All 46 answers chose the same option; the
largest difference in any probability was 0.021. The two servers run the same bf16 weights with
different kernels (the release pads rows to fixed lengths and patches the MPS convolution), so
they agree to bf16 rounding, not bit for bit.

**On the SO-101 kit.** The kit's comparison against hosted Jev 1.14
([docs/so101-kit-checks.md](so101-kit-checks.md)), run on this branch's server: the same skill as
the release's server on 19 of 19 recorded requests and the same pass/stop verdict on 19 of 19
(confidence within 0.013, safety probability within 0.000); 14 of 15 successful steps passed the
gates and all 7 deliberately failed observations stopped. Median time per decision: 1.1 s here,
1.4 s on the release's server.
