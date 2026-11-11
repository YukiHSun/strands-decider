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
| [Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) (`585473d`), 4B | this branch ([docs/mapika-decider.md](mapika-decider.md)); first run on its own server, [Mapika/decider](https://github.com/Mapika/decider) | 14 / 15 | 13 / 15 | 0 | 7 / 7 | 64.6% (its own server) | 1.1 s (1.4 s on its own) |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) (`6cfce5c`), 4B | this branch ([docs/kev.md](kev.md)); first run on its own server, [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (MLX) | 7 / 15 | 7 / 15 | 1 | 7 / 7 | no images | 1.2 s (0.8 s on its own, MLX) |
| [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) (`b3f8a93`), 9B | this branch ([docs/elephant.md](elephant.md)); reference: the release's runtime (OneJev qev + `serve.patch`) on a Colab L4 | 15 / 15 | 14 / 15 | 0 | 7 / 7 | not run | 2.2 s (0.42 s on the L4) |
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
  questions, so those were left out of its eye check. This branch serves it as text only; on the
  kit it gave the same skill and the same pass/stop verdict as the release's server on 19 of 19
  ([docs/mapika-decider.md](mapika-decider.md)).
- **kev-4b** is the fastest, but half of the successful steps fell under the confidence gate,
  and after the lift it chose `transfer` (confidence 0.53) instead of `secure_grasp`, which
  passed. This branch serves it too; on the kit it gave the same skill as the release's server on
  19 of 19 ([docs/kev.md](kev.md)).
- **elephant-1** matched hosted Jev on every count with the kit's wording. This branch gave the
  same skill and verdict as the release's runtime on all 19 steps, all 7 failures and all 19 steps
  with the revised wording ([docs/elephant.md](elephant.md)). It needs a CUDA GPU on its own
  runtime; this branch serves it on a Mac too.
- With the revised wording, clef-flash, decider-4b and elephant-1 all picked `secure_grasp`, a
  skill of the other task, right after the cube-lift task's lift; the kit's wording needs another
  fix there.

The per-request results are in the kit's `outputs/` folder (`decider_*_vs_jev_comparison.json`,
`prompt_order_comparison_strict_anchored_*.json`, `eye_eval_*.json`).

## Which models the `strands-decider` command runs

`strands-decider serve` and `strands-decider ask` load a checkpoint by its kind, so the same
command serves every model below (`--device cuda`, `mps` or `cpu`; `mlx` where noted).

| Model | Command | Notes |
|---|---|---|
| strands-decider (upstream), e.g. `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | `strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mps` | also the gemma4 E2B / E4B / 12B / 26B-A4B builds; `--vision` for images; `--device mlx` on Apple silicon |
| h2oai/h2o-lightning-4b | `strands-decider serve h2oai/h2o-lightning-4b --revision v1.2.3 --attention one-way --noul-floor 0 --device mps` | `--noul-floor 0` keeps raw yes/no probabilities; `--vision` for images; MLX and GGUF builds too ([docs/h2o-lightning.md](h2o-lightning.md)) |
| Cloudflare/clef-flash | `strands-decider serve Cloudflare/clef-flash --device cuda` | `--device mps` on a 32 GB+ Mac; images always on ([docs/clef.md](clef.md)) |
| Mapika/decider-4b | `strands-decider serve Mapika/decider-4b --device mps` | text only (images: its own server, below); yes/no answers at its own temperature 0.05 ([docs/mapika-decider.md](mapika-decider.md)) |
| jaredpalmer/kev-4b | `strands-decider serve jaredpalmer/kev-4b --device mps` | text only; the base `Qwen/Qwen3.5-4B-Base` is downloaded and the LoRA merged at load ([docs/kev.md](kev.md)) |
| baselabs/elephant-1 | `strands-decider serve baselabs/elephant-1 --device mps` | text only; about 19 GB (a 32 GB+ Mac, or a 24 GB GPU with `--device cuda`); block attention read from its `config.json` ([docs/elephant.md](elephant.md)) |

`strands-decider ask <model> --state "..." --noul "..."` answers one question from the command
line with the same loading rules. Point the kit at any of these servers with
`DECIDER_URL=http://127.0.0.1:<port>`.

## Reproducing

Each model runs as a local `POST /v1/systemone` server; the kit's scripts are then pointed at it
with `DECIDER_URL`. The kit compares against hosted Jev, so its `.env` needs `TYPESAFE_API_KEY`.

### The two models this branch serves

```bash
# h2o-lightning-4b v1.2.3, raw yes/no probabilities
strands-decider serve h2oai/h2o-lightning-4b --revision v1.2.3 --attention one-way --noul-floor 0 --device mps --port 8766

# clef-flash, revision fb4417d31c1d0f9e7e77f1141b40aaede7491e00 (bf16, about 19 GB)
strands-decider serve Cloudflare/clef-flash --revision fb4417d31c1d0f9e7e77f1141b40aaede7491e00 --device mps --port 8742
```

### Mapika/decider-4b (its own server)

This branch serves it too (`strands-decider serve Mapika/decider-4b`, text only); its own server is
needed for images and was the reference for this branch's engine.


Revision `585473dec4fa93f0325c2bf02dda7273b61f44c1` (the server reports `decider-4b-v2.1`), about
9 GB with the vision tower.

```bash
git clone https://github.com/Mapika/decider && cd decider
uv venv -p 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install ".[serve]" torchvision   # torchvision: the image processor needs it
huggingface-cli download Mapika/decider-4b --revision 585473dec4fa93f0325c2bf02dda7273b61f44c1 --local-dir ~/models/decider-4b
DECIDER_MODEL=~/models/decider-4b DECIDER_DEVICE=mps DECIDER_IMAGES=1 venv/bin/uvicorn decider.serve:app --host 127.0.0.1 --port 8743
```

`DECIDER_IMAGES=1` turns image input on (the kit's eye check). `/health` reports the per-type
temperatures, including the yes/no temperature of 0.05.

### jaredpalmer/kev-4b (its own server)

This branch serves it too (`strands-decider serve jaredpalmer/kev-4b`); its own server (MLX on a
Mac, or `KEV_BACKEND=torch`) was the reference for this branch's engine.


Revision `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`, a LoRA adapter and head on
`Qwen/Qwen3.5-4B-Base` (revision `1001bb4d`, about 9 GB), which the server fetches from the Hub.

```bash
git clone https://github.com/jaredpalmer/kev && cd kev
uv sync --extra serve        # pins torch < 2.9; on Apple silicon it runs through MLX
python -m kev.serve --run jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52 --host 127.0.0.1 --port 8744 --device mps
```

The kev server has no `/health`; it is ready when it prints `serving jaredpalmer/kev-4b@...`.

### baselabs/elephant-1 (the release's runtime, on a Colab GPU)

This branch serves it (`strands-decider serve baselabs/elephant-1 --revision b3f8a93462fb1051a9de031e94a7ff380e254072 --device mps`).
Its own runtime needs one CUDA GPU; the reference answers came from it on a Colab L4, installed as
the release's `setup.sh` does (OneJev `81ce62f` + the release's `serve.patch`, its pinned
transformers and linear-attention kernels), then replayed to the kit from a small local server.
The model must be loaded from a local directory: qev reads the block attention from
`config.json` only for a local path, and serves a Hub id with plain causal attention.

### The kit's checks

From the root of [pai-final-so101-open-decider](https://github.com/YukiHSun/pai-final-so101-open-decider),
for each server (the port and an output name per model):

```bash
DECIDER_URL=http://127.0.0.1:8742 .venv/bin/python compare_decider_vs_jev.py decider_clef_flash_vs_jev_comparison.json
DECIDER_URL=http://127.0.0.1:8742 .venv/bin/python compare_prompt_order.py strict_anchored clef_flash
# eye check (image models; the server must take images)
H2O_URL=http://127.0.0.1:8743/v1/systemone H2O_TAG=mapika-decider-4b .venv/bin/python eval_eye_models.py h2o
```

### Pitfalls met on the way

- **A stalled Hub download.** `huggingface_hub`'s Xet transfer stopped at 0 KB/s twice here;
  `HF_HUB_DISABLE_XET=1` (plain HTTP) downloaded at about 15 MB/s.
- **The eye check and `box`.** `eval_eye_models.py h2o` also asks `box` questions; decider-4b
  answers none, so they were dropped from the request for it (a wrapper around the script's
  `post`, the kit unchanged).
- **A skill of the other task.** `compare_prompt_order.py` stops when a model returns a skill the
  task does not have; this happened with the revised wording for clef-flash and decider-4b. That
  one request was counted as a wrong, stopped answer (a wrapper around the script's `ask`).
- **elephant-1 on a 24 GB GPU.** qev copies the LM head to float32 on the GPU at load, which
  briefly needs about 6 GB more and ran out of memory on the L4; the same cast done through the
  CPU fits (peak 20.4 GB).
- **A dropped Colab CLI connection.** `colab exec` can wait forever after "Connection was lost";
  the job ran detached on the VM and was polled with short execs.
- **torchvision.** Clef's and decider-4b's own processors load a video processor that needs
  torchvision. This branch's Clef engine uses the PIL image processor instead and does not need it.
