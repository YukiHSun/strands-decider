<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://strandsagents.com/latest/assets/wordmark-github-dark.svg">
        <img src="https://strandsagents.com/latest/assets/wordmark-github-light.svg" alt="Strands" width="320">
      </picture>
    </a>
  </div>

  <h1>Strands Decider</h1>

  <h2>A small, fast decision model for agentic AI.</h2>

  <div align="center">
    <a href="https://github.com/strands-labs/strands-decider/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/strands-labs/strands-decider" alt="Commit Activity"></a>
    <a href="https://github.com/strands-labs/strands-decider/issues"><img src="https://img.shields.io/github/issues/strands-labs/strands-decider" alt="Open Issues"></a>
    <a href="https://github.com/strands-labs/strands-decider/pulls"><img src="https://img.shields.io/github/issues-pr/strands-labs/strands-decider" alt="Open PRs"></a>
    <a href="https://discord.gg/Wa4CQrxsP"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  </div>

</div>

<hr>

> [!NOTE]
> **This fork branch (`feat/clef-flash`) adds five decision models from other vendors, on top of current upstream `main`:**
> [h2oai/h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b),
> [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash),
> [Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b),
> [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) and
> [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1).
> `strands-decider ask` and `strands-decider serve` load each the same way as a strands-decider
> checkpoint, behind the same `POST /v1/systemone` API and with the same answer shape.
>
> ```bash
> strands-decider serve Cloudflare/clef-flash --device cuda        # or --device mps
> strands-decider serve h2oai/h2o-lightning-4b --revision v1.2.3 --attention one-way --noul-floor 0 --device mps
> strands-decider serve Mapika/decider-4b --device mps          # text only
> strands-decider serve jaredpalmer/kev-4b --device mps         # text only
> strands-decider serve baselabs/elephant-1 --device mps        # text only; 9B, a 32 GB+ Mac or a 24 GB GPU
> ```
>
> Every one of them then answers on the same endpoint
> ([how to run them](#running-the-other-vendors-models-with-strands-decider-serve)):
>
> ```bash
> curl -s http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' \
>   -d '{"state": "The door is open.", "questions": {"open": {"type": "noul", "instructions": "Is the door open?"}}}'
> ```
>
> | Model | Served by | What this branch adds | Usage | SO-101 kit: successful steps passed | Wrong skill passed |
> |---|---|---|---|---|---|
> | h2o-lightning-4b (4B) | **this branch** | torch, MLX, images, or a llama-server running its GGUF build; `--noul-floor 0` keeps raw yes/no probabilities | [docs/h2o-lightning.md](docs/h2o-lightning.md) | 15 / 15 | 0 |
> | clef-flash (9B, Apache-2.0) | **this branch** | torch (`cuda`, `mps`, `cpu`), images; answers match the release's reference implementation exactly | [docs/clef.md](docs/clef.md) | 13 / 15 | 1 |
> | Mapika/decider-4b (4B) | **this branch** | torch (`cuda`, `mps`, `cpu`), text only; same skill and verdict as the release's own server on 19 of 19 kit requests | [docs/mapika-decider.md](docs/mapika-decider.md) | 14 / 15 | 0 |
> | jaredpalmer/kev-4b (4B) | **this branch** | torch (`cuda`, `mps`, `cpu`), text only; LoRA merged into its base at load; same skill as the release's own server on 19 of 19 kit requests | [docs/kev.md](docs/kev.md) | 7 / 15 | 1 |
> | baselabs/elephant-1 (9B, MIT) | **this branch** | torch (`cuda`, `mps`, `cpu`), text only; block attention read from its `config.json`; same skill and verdict as the release's own runtime on 19 of 19 kit requests | [docs/elephant.md](docs/elephant.md) | 15 / 15 | 0 |
> | Jev 1.14 (reference) | hosted TypeSafe API | — | — | 15 / 15 | 0 |
>
> All six stopped the kit's 7 deliberately failed observations. Details and the remaining columns
> (revised wording, eye check, time): [docs/so101-kit-checks.md](docs/so101-kit-checks.md).
> Which models the `strands-decider` command itself runs, with a command for each:
> [docs/so101-kit-checks.md#which-models-the-strands-decider-command-runs](docs/so101-kit-checks.md#which-models-the-strands-decider-command-runs).
>
> - The h2o-lightning support is by
>   [yutamorrow](https://github.com/yutamorrow/strands-decider/tree/feat/h2o-lightning-4b),
>   merged onto current `main` in
>   [`feat/h2o-lightning-4b-on-main`](https://github.com/YukiHSun/strands-decider/tree/feat/h2o-lightning-4b-on-main),
>   which adds `--noul-floor` ([details](#this-fork-h2o-lightning-4b-clef-flash-decider-4b-kev-4b-and-elephant-1-on-current-main)). The
>   Clef, decider-4b, kev-4b and elephant-1 support is new in this branch.
> - All five were checked with an SO-101 robot-arm pick-and-place kit,
>   [YukiHSun/pai-final-so101-open-decider](https://github.com/YukiHSun/pai-final-so101-open-decider):
>   15 successful steps of a recorded run must pass every gate, and 7 deliberately failed
>   observations must stop; all five stopped all 7. On its 19 recorded decisions this
>   branch's h2o-lightning also chose the same skill as h2o's own serving on 19 of 19
>   (section 10.6 of the kit's report; Clef: [docs/clef.md](docs/clef.md); decider-4b:
>   [docs/mapika-decider.md](docs/mapika-decider.md); kev-4b: [docs/kev.md](docs/kev.md);
>   elephant-1: [docs/elephant.md](docs/elephant.md)).
> - decider-4b reads its yes/no answers at its own temperature of 0.05, so its safety
>   probabilities are 1.00 or 0.00. kev-4b fell under the confidence gate on half of the steps
>   and let one wrong skill through. elephant-1 matched hosted Jev on every count. All six side by side,
>   with how each was run (commands, revisions, pitfalls):
>   [docs/so101-kit-checks.md](docs/so101-kit-checks.md).

Strands decider is one of a new class of **decision models**, or "system one" models. Unlike an
LLM, which can generate arbitrary text, a decision model picks between sets of options and
rates things on a scale. This class of model works best for problems that fall between LLMs
and traditional classification models: it is a general-purpose classification and scoring
model that responds faster than an LLM and does not take the time and expertise to train
that a traditional classifier does. That makes it a natural fit for the decisions inside
agentic workflows built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

- **Pick from options** — "Are there three r's in strawberry? Yes or no." "What language is
  the phrase 'sihamba ngokushesha' in? English, Zulu, or Dutch."
- **Rate on a scale** — "How positive is 'this is the best doc I've ever read'? Between 0 and 1."
- **Calibrated reliability scores** — every decision carries a confidence. On short
  classification tasks it has never seen, answers at a confidence of 0.9 or more are right
  about 95% of the time ([evaluation/results.md](evaluation/results.md#summary)); below that,
  confirm or ask a person. Frontier LLM inference APIs do not expose anything equivalent.

## How to start using strands-decider

Start on your own machine: install the `strands-decider` CLI, ask the model questions, then run it
as a server and send it HTTP requests. When other people or services need to call it, deploy that
server ([Deploying it](#deploying-it)).

The easiest place to get started is through the `strands-decider` cli:

```bash
pip install strands-decider
```

On an Apple-silicon Mac, `--device mlx` runs the model through MLX, 1.4 to 1.6x faster than MPS.
It needs the `mlx` extra, which ships with the next release; until then, install from a clone with
`pip install -e ".[mlx]"`. The `cuda`, `mps` and `cpu` extras name the other devices
([docs/inference.md](docs/inference.md#environments-for-serving)).
The example outputs below are from `strands-decider-2B-qwen3.5-v1-2610`, recorded with
`--device cpu`; other devices differ in the last digits.

### Choice question
You can ask the model to choose based on some state and a question:
```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" 
```
Example Output:

```bash
choice_0 -> billing (confidence 0.659)
  billing                  0.773
  sales                    0.121
  retail                   0.106
```

### Noul question
You can also ask the model a Yes/No question:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --noul "Does this convey urgency?" 
```

Example Output:

```bash
noul_0 noul = 0.858
```
Closer to 1 is leaning more toward Yes

### Score question
Or give a question a score:
```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
Example Output:

```bash
score_0 score = 1.03 (confidence 0.509)
  0: calm                                     0.196
  1: frustrated                               0.574
  2: depressed                                0.230
```

### Multiple question types

You can combine multiple questions into a single command. This is more efficient as you only need to load the state for the model once:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
<details>
  <summary>Example Output</summary>

  ```bash
  noul_0 noul = 0.858
  choice_0 -> billing (confidence 0.659)
    billing                  0.773
    sales                    0.121
    retail                   0.106
  score_0 score = 1.03 (confidence 0.509)
    0: calm                                     0.196
    1: frustrated                               0.574
    2: depressed                                0.230
  ```
</details>

### Running as a server

You can also run the model as a server, and ask questions via http requests:
```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
```

```bash
curl -s localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
    }
  }'
```

<details>
  <summary>Example Output</summary>

  ```bash
  {
    "model": "strands-decider-2B-qwen3.5-v1-2610",
    "answers": {
      "is_urgent": {
        "type": "noul",
        "noul": 0.8581
      }
    },
    "usage": {
      "input_tokens": 86,
      "output_tokens": 1
    },
    "latency_ms": 489.01
  }          
  ```
</details>

`ask` and `serve` also read `h2oai/h2o-lightning-4b` checkpoints ([docs/h2o-lightning.md](docs/h2o-lightning.md)).

#### This fork: h2o-lightning-4b, clef-flash, decider-4b, kev-4b and elephant-1 on current main

The h2o-lightning-4b support was written by
[yutamorrow](https://github.com/yutamorrow/strands-decider/tree/feat/h2o-lightning-4b)
(`feat/h2o-lightning-4b`). Upstream keeps this repository for strands-decider models
([issue #95](https://github.com/strands-labs/strands-decider/issues/95)), so it lives in forks.
This branch merges that work onto current `main` (the engine built in the server's lifespan,
`--device-map`, `--max-concurrency`, `--warmup`) and adds `--noul-floor`.

Checked against a robot-arm decision task: the 19 recorded requests of an SO-101 pick-and-place
kit ([pai-final-so101-open-decider](https://github.com/YukiHSun/pai-final-so101-open-decider)),
sent to `strands-decider serve h2oai/h2o-lightning-4b --revision v1.2.3 --attention one-way
--noul-floor 0 --device mps`, chose the same skill as h2o's own serving (community MLX build,
8-bit) on 19 of 19, with the same pass/stop verdict on all 19. The choice confidence differed by
at most 0.029 and the safety yes/no probability by at most 0.009. All 7 deliberately failed
observations were stopped.

The clef-flash support is new in this branch (`feat/clef-flash`, on top of
`feat/h2o-lightning-4b-on-main`). `src/strands_decider/clef.py` ports the release's prompt layout
and joint schema head from its reference implementation (Apache-2.0); no code from the model repo
is run. On 27 requests (the kit's 26 and one with an image) its answers were identical to the
reference `systemone()` on the same weights. Usage and the kit results: [docs/clef.md](docs/clef.md).

The Mapika/decider-4b support is new in this branch too. `src/strands_decider/mapika.py` ports the
rendering and readout of the release's own server (Apache-2.0); it serves text only. On the SO-101
kit it chose the same skill and gave the same pass/stop verdict as the release's server on 19 of
19 requests. Usage and the check: [docs/mapika-decider.md](docs/mapika-decider.md).

The jaredpalmer/kev-4b support is new in this branch as well. `src/strands_decider/kev.py` ports
the release's rendering, packing and pointer-head readout (Apache-2.0) and merges its LoRA into
the base at load; it serves text only. On the SO-101 kit it chose the same skill as the release's
server on 19 of 19 requests. Usage and the check: [docs/kev.md](docs/kev.md).

The baselabs/elephant-1 support is new in this branch as well. `src/strands_decider/elephant.py`
ports the prompt rendering, block attention and readout of the runtime the release names (OneJev
`qev` with the release's `serve.patch`, Apache-2.0); it serves text only. The release's runtime
needs a CUDA GPU, so it was run on a Colab L4 as the reference: 67 of 67 answers chose the same
option (largest probability difference 0.026), and on the SO-101 kit it gave the same skill and
verdict on 19 of 19 requests. Usage and the check: [docs/elephant.md](docs/elephant.md).

The six models side by side (these five and hosted Jev), with the commands, revisions and
pitfalls to repeat each run: [docs/so101-kit-checks.md](docs/so101-kit-checks.md).

#### Running the other vendors' models with `strands-decider serve`

`strands-decider serve` recognises each checkpoint by its files and loads the matching engine, so
the same command and the same API serve all of them. The first start downloads the weights from
the Hugging Face Hub (or pass a local folder instead of the repo id).

```bash
# install this branch
git clone -b feat/clef-flash https://github.com/YukiHSun/strands-decider && cd strands-decider
pip install -e .    # or: uv pip install -e .

# start one server (--port defaults to 8000; --device cuda / mps / cpu)
strands-decider serve h2oai/h2o-lightning-4b --revision v1.2.3 --attention one-way --noul-floor 0 --device mps
strands-decider serve Cloudflare/clef-flash --device cuda        # 9B; --device mps on a 32 GB+ Mac
strands-decider serve Mapika/decider-4b --device mps
strands-decider serve jaredpalmer/kev-4b --device mps
strands-decider serve baselabs/elephant-1 --device mps --port 8748

# check which engine it loaded
curl -s http://127.0.0.1:8748/health

# ask it (one request, any number of noul / choice / score questions)
curl -s http://127.0.0.1:8748/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}}
  }
}'

# or one question without a server
strands-decider ask baselabs/elephant-1 --device mps --state "The door is open." --noul "Is the door open?"
```

| Model | Size on disk | Devices | Images | Notes |
|---|---|---|---|---|
| h2oai/h2o-lightning-4b | about 9 GB | cuda, mps, cpu, mlx (light builds), GGUF via llama-server | `--vision` | `--noul-floor 0` for raw yes/no probabilities ([docs/h2o-lightning.md](docs/h2o-lightning.md)) |
| Cloudflare/clef-flash | about 19 GB | cuda, mps, cpu | always on | answers match the release's reference exactly ([docs/clef.md](docs/clef.md)) |
| Mapika/decider-4b | about 8.5 GB | cuda, mps, cpu | no | yes/no answers at its own temperature 0.05 ([docs/mapika-decider.md](docs/mapika-decider.md)) |
| jaredpalmer/kev-4b | adapter + about 9 GB base | cuda, mps, cpu | no | the base is downloaded and the LoRA merged at load ([docs/kev.md](docs/kev.md)) |
| baselabs/elephant-1 | about 19 GB | cuda (24 GB is enough), mps (32 GB+ Mac), cpu | no | block attention from its `config.json`; mask-based attention only (sdpa or eager) ([docs/elephant.md](docs/elephant.md)) |

To run the SO-101 kit against any of them, point it at the server:
`DECIDER_URL=http://127.0.0.1:<port>` ([docs/so101-kit-checks.md](docs/so101-kit-checks.md#reproducing)).

### Images

Qwen3.5-2B-Base is natively multimodal. With `--vision` the server keeps its vision tower,
and a request may carry `images` (base64) as part of the state; the published checkpoints
(v1, and hobson-v21 and v19 before it) answer over them with no image training. Needs `pip install "strands-decider[vision]"` and
transformers 5.18 or later.

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --vision --port 8000
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --state "" --image page.png \
  --noul "Is the signature block filled in?"
```

v19 matches an image-trained 2B decider on accuracy, and on NaturalBench is much better
calibrated (ECE 0.014 against 0.080). v21 has v19's image accuracy, is better calibrated on
POPE, and is more confident than v19 when the image is missing. v1 has v21's image accuracy;
with the image missing it is more overconfident than v21 on POPE
([docs/vision.md](docs/vision.md#v1)). See
[docs/vision.md](docs/vision.md) for the request shape, how it works and the measurements.

### Deploying it

The server you ran above is what you deploy. To host it beyond your machine, see
[deploy/README.md](deploy/README.md): it lists where the server can run and links to a guide for
each.

## Model names

Model names now use the pattern `strands-decider-<size>-<base>-v<N>-<YYMM>`:

- `<size>` is the size of the base model, as its Hugging Face id writes it (`2B`, `E4B`).
- `<base>` is the base model family (`qwen3.5`, `gemma4`).
- `v<N>` counts the releases on one base family: `v1` is the first release on each base.
  Releases with the same `N` on different bases can use different recipes; each model card says
  what its model trained on.
- `<YYMM>` is the release month.

Examples: `strands-decider-2B-qwen3.5-v1-2610` (Qwen3.5) and `strands-decider-E4B-gemma4-v1-2610`
(Gemma 4). The earlier models keep their names: `strands-decider-2B-hobson-v19` and
`strands-decider-2B-hobson-v21`. A second release in the same month updates the same Hub
repository and adds a revision tag `YYMMDD`. [docs/naming.md](docs/naming.md) has the full rules.

The models live in the [`StrandsAgents`](https://huggingface.co/StrandsAgents) organization on
Hugging Face, their main home. The five v1 models are also in the
[`amazon`](https://huggingface.co/amazon) organization, with the same files and the same names:
for example `amazon/strands-decider-E4B-gemma4-v1-2610`.

## Gemma 4 models

Four models on the Gemma 4 instruction-tuned base models (`google/gemma-4-*-it`, Apache-2.0). They
have the same pointer head and answer the same three question types as the Qwen3.5 model.

| model | JevBench (231) | Brier | held-out short tasks | held-out code tasks | Decision Index | GPU memory for the weights (bf16) | median latency per JevBench question, one H100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610) | 180 | 0.313 | 0.652 | 0.773 | 29.78 | about 9 GB | 102 ms |
| [`strands-decider-E4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610) | 187 | 0.243 | 0.684 | 0.815 | 39.42 | about 15 GB | 139 ms |
| [`strands-decider-12B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610) | 200 | 0.173 | 0.708 | 0.870 | 47.14 | about 23 GB | 136 ms |
| [`strands-decider-26B-A4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610) | 207 | 0.162 | 0.708 | 0.859 | 48.69 | about 51 GB | 245 ms |

For comparison, `strands-decider-2B-qwen3.5-v1-2610` scores 180, Brier 0.280. The E2B Decision Index
was measured before the E2B attention fix (see [docs/inference.md](docs/inference.md#gemma-4-models)),
so it may be off. Every model is a soup of three training runs, calibrated afterwards
([evaluation/results.md](evaluation/results.md#gemma-4-v1)).

**Choose a size.** A larger model is more accurate and better calibrated, and needs more memory.
The 12B and the 26B-A4B are the most accurate; the 26B-A4B is better calibrated, but needs about
twice the memory and twice the latency of the 12B. The E4B is a good choice when the model
must share one GPU with other work. The E2B is the smallest: use it where memory is tightest, or on
CPU. All four serve on CPU ([docs/inference.md](docs/inference.md#gemma-4-models)).

**Install.** Gemma 4 support is on `main` and not yet in the package on PyPI:

```bash
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
strands-decider ask StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

<details>
  <summary>Example Output (`--device cpu`)</summary>

  ```bash
  noul_0 noul = 0.959
  choice_0 -> billing (confidence 0.951)
    billing                  0.967
    sales                    0.019
    retail                   0.013
  score_0 score = 0.97 (confidence 0.848)
    0: calm                                     0.102
    1: frustrated                               0.823
    2: depressed                                0.075
  ```
</details>

**What they train on.** The Gemma 4 models do not use the Qwen3.5 v1 data mix. They train on
hobson-v21's data (123,339 rows) plus 12,263 code-task rows, 135,602 rows in all. Qwen3.5-4B is the
teacher, where its answer agrees with the gold label, and training keeps the answers close to the
released v19. The rows come from 25 public Hugging Face datasets, five of them code datasets, and from
ContractNLI, MuSiQue, generated rows and generated code tasks. [data/sources.md](data/sources.md)
lists every source with its licence.

The Gemma 4 models read text only: `--vision` is for the Qwen3.5 model.

## About the model

`strands-decider-2B`, the first model of the family, has 1.9 billion parameters. It answers a
question in a median 115 ms on an RTX 3090 and also serves on an Apple-silicon Mac or on CPU,
and many questions about one text are cheap, because the text is read once and each question
adds only its own tokens
([docs/inference.md](docs/inference.md#asking-many-questions-is-nearly-free)). Its
accuracy and calibration are measured on JevBench, a third-party benchmark for this class of
model ([Performance](#performance)).


## Model Architecture

The core idea: take a pretrained decoder LLM torso ([Qwen3.5-2B-Base](https://huggingface.co/Qwen)),
**discard its language-modelling head** — taking away its ability to generate text — and
replace it with a small **pointer head** of about a million parameters. The head scores each
option by comparing the hidden state at the `<answer>` position against the hidden state at
that option's own last token. One forward pass, no generation, no decoding loop. The torso is
adapted with a rank-16 LoRA adapter, and the head runs in fp32.

Because the head holds no per-option parameters, nothing can learn that "the first option is
usually right", nothing caps how many options a question may carry, and label sets are
defined by the request rather than baked into the weights. Three question types come out of
the same masked softmax, read back differently: `noul` (yes/no), `choice` (one of N) and
`score` (an ordered rubric).

As you browse the research, you will find that this is the second major iteration of the
architecture. The first used a slot head, which mapped the final hidden state to a fixed set
of slots and performed significantly worse. Every change since is captured in the research
so you can follow along with the work. The reference model today is **v1**
(`strands-decider-2B-qwen3.5-v1-2610`; [docs/naming.md](docs/naming.md) explains the name).
It trains on a new, balanced data mix of 246,678 rows, about twice v21's: it adds code tasks,
new rule-application rows (apply a written policy to a case) and rows from more public datasets,
so that the mix follows the kinds of decisions in the evaluations. Its teacher is
gemma-4-31B-it, and training keeps its answers close to the released v19. The released
checkpoint is a soup of three training runs: their LoRA updates and head scores averaged
exactly, then calibrated
([evaluation/results.md](evaluation/results.md#v1-the-released-model)).
`strands-decider-2B-hobson-v21` and `-v19` stay published.
[docs/architecture.md](docs/architecture.md#the-architecture) has the full design, the
training objective, and the decisions behind them.

## Performance

Three targets matter: **accuracy**, **calibration** and **latency**. v1, v21 and v19 measure:

| Measurement | v1 | v21 | v19 | Source |
| --- | --- | --- | --- | --- |
| JevBench v1 public set, 231 tasks, accuracy | 0.779 (180 of 231); its three seeds 179.7 ± 3.8 | 0.762 (176 of 231); six-seed mean 0.758 (175.0, SD 2.4) | 0.723 (167 of 231) | [evaluation/results.md](evaluation/results.md#v1-the-released-model) |
| JevBench Brier score / expected calibration error | 0.280 / 0.072 | 0.323 / 0.064 | 0.342 / 0.052 | [evaluation/README.md](evaluation/README.md) |
| Tiers (this repository's split of the public tasks): easy / standard / hard | 1.000 / 0.931 / 0.586 | 1.000 / 0.931 / 0.550 | 1.000 / 0.875 / 0.505 | [evaluation/jevbench.md](evaluation/jevbench.md#board-position-v142-25-september-2026) |
| Latency per JevBench question, RTX 3090 under WSL2, median / 95th percentile | same architecture and size as v19 | same architecture and size as v19 | 115 ms / 299 ms | [evaluation/results.md](evaluation/results.md#summary) |
| Latency per question, M3 Pro (Apple silicon), warm median, under 300 tokens / all tasks | same architecture and size as v19 | same architecture and size as v19 | 153 ms / 234 ms | [evaluation/results.md](evaluation/results.md#serving-on-a-mac-accuracy-and-latency) |

Every task in the easy tier is answered correctly, and the nearest comparison is
`decider-2b`, which shares v19's torso with a different recipe
([evaluation/jevbench.md](evaluation/jevbench.md#against-the-nearest-open-systems)).

v1's column is measured through `main` (the training harness gives 181; one task is a near tie).
v1 also answers code questions: 0.836 on 1,996 held-out code tasks
([evaluation/results.md](evaluation/results.md#v1-the-released-model)).

Read the v21 and v19 columns as two single runs, not as a measured gain. Trained on one
host with six seeds each, the v19 recipe and v21's average 172.8 and 172.3 JevBench tasks:
the same accuracy. v21's Brier score is lower on that host (0.331 against 0.341), and the
two recipes differ by the paraphrases and the distillation only.

v19's JevBench figures are at the 3072-token window the run was preregistered at; the recipe
saves a 4096-token window, at which v19 scores 168. And 231 tasks are few: six retrains of
the v17 recipe had a standard deviation of 3.2 tasks, so treat a difference under about 10
tasks between two single runs as unresolved. [evaluation/README.md](evaluation/README.md)
has the scripts and the limitations, and links the results by version and the board
position with its caveats.

## Why 2B?

Two reasons. First, **experimentation**: at 1.9 billion parameters you can serve the model,
and retrain the whole recipe, on hardware you already have — about 11 hours on one RTX 3090,
and serving works on an Apple-silicon Mac. That makes trying an idea fast and low-risk.
Second, ~2B parameters looks like a sweet spot: small enough to experiment with, large enough
to do meaningful work.

## What can I do with it?

The decisions inside an agentic workflow are the natural target:

- **Model routing** — pick the right LLM for a task
- **Tool selection** — decide which tool an agent should call next
- **Argument checking** — verify a tool call's arguments before it runs
- **Triage** — route an incoming request to the team or queue that should own it
- **Guardrails** — grounding checks, safety classification, policy classification
- **Evaluations** — score model outputs, and check whether an answer is adequate, at low cost
- **Hybrid agents** — let the LLM make the hard decisions and a decider make the rote ones,
  reducing cost and latency

## Trying it in an Agent

The repository includes a worked example of strands decider inside a Strands agent, under
[`examples/strands/`](examples/strands/README.md): a `before_tool_call` intervention that gates a
weather-tool call on two yes/no decisions, so the agent asks which city instead of guessing. See
the example's [README](examples/strands/README.md) for setup and the walk-through.

## Customizing for your decisions

[`skills/customize-decider/`](skills/customize-decider/SKILL.md) is a skill for an AI agent.
It guides you from a decision in your application to a model measured on your own data: the task
and its questions, an evaluation set, scores of every released model, the decision whether to
train, the training data, the run and its monitoring, calibration, and the comparison with the
released model.

The skill is plain Markdown: a `SKILL.md` with a short front matter, and reference files that it
loads one phase at a time. It works with any agent that can read files and run shell commands, such
as an agent built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) or
a coding CLI. If your agent supports skills, install the folder where it looks for skills. If not,
tell the agent to read `skills/customize-decider/SKILL.md` and follow it. Then ask it to evaluate
Strands Decider on your task.

## Benchmarking inference performance

[`benchmark/`](benchmark/README.md) measures how a serving change affects the configured models in
one compute environment. It runs the same synthetic workload against an accepted baseline and a
candidate, then records peak requests per second, concurrency-1 latency, optional cost per
1 million decisions, and a separate reliability gate. It benchmarks managed local servers or an
already-running HTTP endpoint:

```bash
python -m benchmark.runner run local
python -m benchmark.runner run local MODEL  # one model only
python -m benchmark.runner compare local MODEL CANDIDATE_DIR
```

Full runs also create environment-level throughput and latency graphs with one line per model.

## Training

Two entry points, both on a Linux or WSL2 host with NVIDIA GPUs and the training
environment of [training/README.md](training/README.md#setup) (the `train` extra, the
pinned torch and `flash-linear-attention`):

- `training/recipe.sh all`: one host, local, 1 to 8 GPUs (`NGPU=8` for eight). It builds
  the corpora, trains, calibrates and evaluates.
- `training/run_recipe.sh all` with [training/aws/](training/aws/README.md): a distributed
  8-GPU host. It adds per-stage timing and logs, row-count checks, the S3 copy of the
  outputs (`PY` and `S3_PREFIX` must be set), and `FAST=1`, which trades 24 GB
  compatibility for speed on 80 GB GPUs.

About 11 hours on one RTX 3090 (24 GiB), or 1 hour 10 minutes on eight H100s with `FAST=1`.
[training/README.md](training/README.md) has the setup, the stages, the settings and the
hardware notes; [data/sources.md](data/sources.md) lists every source and its licence, and
[data/README.md](data/README.md#reproduction-contract) states the reproduction contract.

## The research

The record of the work is in the repository. Since v9, every training run states its
predictions and its failure conditions before training, and the outcome is appended after
the run without editing what came before. By that rule, a run that misses its bar does not
replace the reference model; five runs were promoted by the maintainers' decision anyway,
and the record says so. Most runs missed their bar, v20 among them: 169 of 231 at the
4096-token window against v19's 168, inside the retrain noise, with four predictions
failed. [research/README.md](research/README.md) lists each run and its outcome, and
[research/history.md](research/history.md) tells what moved the benchmark and what did
not. To propose an experiment, open an issue with a preregistration ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Community

To ask questions and talk about Strands Decider, join the [Strands Decider channel on Discord](https://discord.gg/Wa4CQrxsP).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
