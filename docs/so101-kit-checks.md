# Decision models on an SO-101 robot-arm task

The models this branch serves, and two more small open models served by their authors' own
servers, were run on the same checks: the decision scripts of the SO-101 pick-and-place kit
[YukiHSun/pai-final-so101-open-decider](https://github.com/YukiHSun/pai-final-so101-open-decider)
(sections 10.6 to 10.8 of its report), unchanged, against each model's `POST /v1/systemone`.

The kit asks for the next skill of a cube-lift or pick-and-place task and stops on a confidence
gate (0.45), a safety yes/no question (0.60 or more) and a transition check. The checks:

- 15 successful steps of a recorded run: how many pass every gate with the recorded skill, with
  the kit's wording and with its revised wording (`strict_anchored`);
- how many wrong skills still pass every gate (the dangerous case);
- 7 deliberately failed observations: how many stop;
- for image models, 15 camera images read as the kit's "eye" (agreement with Claude's labels).

All were run on an Apple M-series Mac (36 GB), 2026-11-10 and 11-11; hosted Jev 1.14 is the
reference.

| Model | Served by | Passed, kit wording | Passed, revised wording | Wrong skill passed | Failures stopped | Eye (15 images) | Median time |
|---|---|---|---|---|---|---|---|
| [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b) v1.2.3, 4B | this branch (`--attention one-way --noul-floor 0`) | 15 / 15 | 15 / 15 | 0 | 7 / 7 | 83.8% (v1.4.2) | 1.1 s |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9B | this branch ([docs/clef.md](clef.md)) | 13 / 15 | 12 / 15 | 1 | 7 / 7 | not run | 1.8 s |
| [Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) (`585473d`), 4B | its own server, [Mapika/decider](https://github.com/Mapika/decider) | 14 / 15 | 13 / 15 | 0 | 7 / 7 | 64.6% | 1.4 s |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) (`6cfce5c`), 4B | its own server, [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (MLX) | 7 / 15 | 7 / 15 | 1 | 7 / 7 | no images | 0.8 s |
| Jev 1.14 (hosted) | TypeSafe API | 15 / 15 | 15 / 15 | 0 | 7 / 7 | 84.6% | 0.15 s |

Notes:

- **h2o-lightning-4b** keeps the yes/no gate useful only with its floor off (`--noul-floor 0`);
  with the checkpoint's 0.801 floor an unsure yes passes the 0.60 gate.
- **clef-flash** stops every failure with raw probabilities (no floor). Its wrong pass was
  `retreat` chosen again after `retreat` instead of `done`. This branch's engine matches the
  release's reference implementation exactly (docs/clef.md).
- **decider-4b** reads its yes/no answers at temperature 0.05 (its `decider_config.json`), so the
  safety probabilities are 1.00 or 0.00: every failure stopped here, but a wrong yes would be
  just as certain. As an eye it often said the cube was not visible. It answers no `box`
  questions, so those were left out of its eye check. Not added to this branch.
- **kev-4b** is the fastest, but half of the successful steps fell under the confidence gate,
  and after the lift it chose `transfer` (confidence 0.53) instead of `secure_grasp`, which
  passed. Not added to this branch.
- With the revised wording, clef-flash and decider-4b both picked `secure_grasp`, a skill of the
  other task, right after the cube-lift task's lift; the kit's wording may need another fix there.

The per-request results are in the kit's `outputs/` folder (`decider_*_vs_jev_comparison.json`,
`prompt_order_comparison_strict_anchored_*.json`, `eye_eval_*.json`).
