"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .clef import ClefEngine, load_clef_engine
from .clef import resolve as resolve_clef
from .elephant import ElephantEngine, load_elephant_engine
from .elephant import resolve as resolve_elephant
from .h2o import H2OEngine, H2OEngineConfig, load_h2o_engine, resolve
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .kev import KevEngine, load_kev_engine
from .kev import resolve as resolve_kev
from .mapika import MapikaEngine, load_mapika_engine
from .mapika import resolve as resolve_mapika
from .modeling import StrandsDeciderModel, checkpoint_dir
from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion when two overlapping
    threads hit the same device. CUDA and CPU are thread-safe for inference, and the MLX
    engine runs every evaluation on a single worker thread."""
    return str(device).startswith("mps")


def warm_up(
    engine: SystemOneEngine | H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine,
    max_batch: int,
    vision: bool = False,
) -> None:
    """Ask every question kind, then every number of questions up to `max_batch`, once.

    On CUDA the first forward of a shape compiles kernels (Triton's, on a Qwen3.5 torso), once
    for each number of questions, whatever the state's length. A request with more than
    `max_batch` questions runs in chunks of at most `max_batch`, so these shapes cover every
    request after this. With `vision`, one image request too: see `_warm_up_image`.
    """
    state = "A warm-up request, answered before the server accepts requests."
    kinds: dict[str, Question] = {
        "noul": NoulQuestion(instructions="Is this a warm-up?"),
        "choice": ChoiceQuestion(instructions="Pick one.", criteria={"a": None, "b": None}),
        "score": ScoreQuestion(instructions="How warm is it?", criteria=["cold", "warm"]),
    }
    started = time.perf_counter()
    for name, question in kinds.items():
        engine.evaluate(SystemOneRequest(state=state, questions={name: question}))
    engine.evaluate(SystemOneRequest(state=state, questions=kinds))
    for n in range(2, max_batch + 1):
        questions: dict[str, Question] = {f"q{i}": kinds["noul"] for i in range(n)}
        engine.evaluate(SystemOneRequest(state=state, questions=questions))
    if vision:
        _warm_up_image(engine, state, kinds["noul"])
    print(f"[strands-decider] warmed up in {time.perf_counter() - started:.1f} s")


def _warm_up_image(
    engine: SystemOneEngine | H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine,
    state: str,
    question: Question,
) -> None:
    """One image request: the first one initialises the image path; after it, image requests
    reuse the text path's kernels, whatever the number of questions or images. The image is
    small, so the request fits any window.
    """
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    engine.evaluate(SystemOneRequest(state=state, questions={"q": question}, images=[image]))


def _load_h2o(
    path: str,
    *,
    device: str,
    attention: str | None,
    model_name: str,
    attn_implementation: str | None,
    vision: bool = False,
) -> H2OEngine:
    if device == "mlx":
        from .h2o_mlx import load as load_h2o_mlx

        return load_h2o_mlx(path, attention, H2OEngineConfig(device="mlx", model_name=model_name))
    return load_h2o_engine(
        path,
        device=device,
        attention=attention,
        attn_implementation=attn_implementation,
        model_name=model_name,
        vision=vision,
    )


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    device_map: str | None = None,
    warmup: bool = True,
    build_engine: Callable[
        [], SystemOneEngine | H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine
    ]
    | None = None,
    max_concurrency: int | None = None,
    attention: str | None = None,
    revision: str | None = None,
    engine: str = "local",
    llama_server_url: str | None = None,
    llama_server_timeout: float = 60.0,
    noul_floor: float | None = None,
) -> FastAPI:
    """The app for `checkpoint`. An h2o-lightning checkpoint (see h2o.py) is read with
    `attention` ("two-way" or "one-way"; None: the build's default, two-way for the HF build,
    one-way for the light builds); `attention` is not used for a strands-decider checkpoint.
    `revision` is the Hub revision to download (None: the default branch). `vision` keeps the
    vision tower, on torch devices only. `engine="llama-server"` answers through a llama-server
    running h2o-lightning's GGUF build at `llama_server_url` (see `infer.load_engine`); its
    failures are HTTP 502."""
    from .h2o import ATTENTION_MODES, ENGINES, MLX_IMAGES_UNSUPPORTED, check_noul_floor
    from .h2o_llama import LlamaServerError
    from .infer import MLX_EXTRA_MESSAGE, load_engine, mlx_available

    if attention is not None and attention not in ATTENTION_MODES:
        raise ValueError(f"attention must be one of {list(ATTENTION_MODES)}, got {attention!r}")
    engine_name = engine
    floor = check_noul_floor(noul_floor)
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}, got {engine!r}")
    if device_map and (vision or not device.startswith("cuda")):
        raise ValueError("--device-map splits the text torso across CUDA GPUs: use it with cuda")
    if vision and device == "mlx" and engine == "local":
        # Checked before anything loads, so it is this error with or without the mlx extra.
        raise ValueError(
            f"--vision runs on torch devices (cuda, mps, cpu), not mlx: {MLX_IMAGES_UNSUPPORTED}"
        )
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )

    def _default_build_engine() -> (
        SystemOneEngine | H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine
    ):
        if engine == "llama-server":
            llama = load_engine(
                checkpoint,
                attention=attention,
                revision=revision,
                engine=engine,
                llama_server_url=llama_server_url,
                llama_server_timeout=llama_server_timeout,
            )
            assert isinstance(llama, H2OEngine)
            llama.cfg.model_name = resolved_name
            return llama
        h2o_path = resolve(checkpoint, revision, refuse_quantised=device != "mlx")
        if h2o_path is not None:
            if device == "mlx" and not mlx_available():
                raise RuntimeError(MLX_EXTRA_MESSAGE)
            return _load_h2o(
                h2o_path,
                device=device,
                attention=attention,
                model_name=resolved_name,
                attn_implementation=attn_implementation,
                vision=vision,
            )
        clef_path = resolve_clef(checkpoint, revision)
        if clef_path is not None:
            if device == "mlx" or device_map:
                raise ValueError(
                    "a Clef checkpoint runs on one torch device (cuda, mps or cpu): "
                    "--device mlx and --device-map are not supported for it"
                )
            return load_clef_engine(
                clef_path,
                device=device,
                model_name=resolved_name,
                attn_implementation=attn_implementation,
            )
        mapika_path = resolve_mapika(checkpoint, revision)
        if mapika_path is not None:
            if device == "mlx" or device_map or vision:
                raise ValueError(
                    "a Mapika decider checkpoint runs on one torch device (cuda, mps or cpu), "
                    "text only: --device mlx, --device-map and --vision are not supported for it"
                )
            return load_mapika_engine(
                mapika_path,
                device=device,
                model_name=resolved_name,
                attn_implementation=attn_implementation,
            )
        kev_path = resolve_kev(checkpoint, revision)
        if kev_path is not None:
            if device == "mlx" or device_map or vision:
                raise ValueError(
                    "a Kev checkpoint runs on one torch device (cuda, mps or cpu), text only: "
                    "--device mlx, --device-map and --vision are not supported for it"
                )
            return load_kev_engine(
                kev_path,
                device=device,
                model_name=resolved_name,
                attn_implementation=attn_implementation,
            )
        elephant_path = resolve_elephant(checkpoint, revision)
        if elephant_path is not None:
            if device == "mlx" or device_map or vision:
                raise ValueError(
                    "a qev checkpoint (elephant-1) runs on one torch device (cuda, mps or cpu), "
                    "text only: --device mlx, --device-map and --vision are not supported for it"
                )
            return load_elephant_engine(
                elephant_path,
                device=device,
                model_name=resolved_name,
                attn_implementation=attn_implementation,
            )
        source = checkpoint if revision is None else checkpoint_dir(checkpoint, revision)
        if vision:
            from .vision import load_vision_engine

            return load_vision_engine(source, config, attn_implementation=attn_implementation)
        if device == "mlx":
            return load_mlx(source, config)
        model = StrandsDeciderModel.load(
            source, device_map=device_map, attn_implementation=attn_implementation
        )
        return SystemOneEngine(model, config)

    # Tests inject a stub via `build_engine=...` to skip model loading. Construction
    # still runs under the lifespan, so a half-built engine is still closed.
    build = build_engine or _default_build_engine

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build()
        if isinstance(engine, H2OEngine) and floor is not None:
            # --noul-floor: the yes/no floor of an h2o-lightning engine, whatever its backend.
            engine.cfg.noul_floor = floor
        app.state.engine = engine
        # None: requests share anyio's default worker pool (40 threads), as sync handlers do.
        app.state.limiter = anyio.CapacityLimiter(max_concurrency) if max_concurrency else None
        try:
            # Listener binds only after lifespan startup returns, so /health (/ping)
            # fails until this is done.
            if warmup:
                # An h2o-lightning engine forwards each question on its own, so a request's
                # shape does not depend on how many questions it has: the three kinds suffice.
                # A llama-server warms itself: warming it from here would only send it requests.
                if engine_name != "llama-server":
                    # A Clef engine answers a request in one forward of one record, whatever
                    # its number of questions: the three kinds suffice there too.
                    one = isinstance(
                        engine, H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine
                    )
                    warm_up(engine, 1 if one else max_batch, vision=vision)
            yield
        finally:
            # The MLX engine owns a worker thread (see mlx_engine.MLXEngine); other engines
            # have nothing to close.
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
        lifespan=lifespan,
    )

    engine_lock = threading.Lock()
    guard: contextlib.AbstractContextManager
    guard = engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    # async, like /ping below: FastAPI runs a sync dependency in the worker thread pool too.
    # Any, not SystemOneEngine | H2OEngine: FastAPI evaluates these annotations when the app is
    # built, and tests replace SystemOneEngine with a stub function.
    async def get_engine(request: Request) -> Any:
        engine = request.app.state.engine
        return engine

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it. /ping is async: a sync handler would wait for a free
    # thread in the pool that /invocations fills (40 threads), and SageMaker allows a
    # health check 2 s.
    @app.get("/health")
    @app.get("/ping")
    async def health(eng: Any = Depends(get_engine)) -> dict:
        fields = (
            eng.health_fields()
            if isinstance(eng, H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine)
            else {
                "base_model": eng.model.config.base_model,
                "max_length": eng.model.config.max_length,
                "temperature": eng.model.config.temperature,
            }
        )
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            **fields,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
        }

    def evaluate(
        eng: SystemOneEngine | H2OEngine | ClefEngine | MapikaEngine | KevEngine | ElephantEngine,
        request: SystemOneRequest,
    ) -> SystemOneResponse:
        with guard:
            return eng.evaluate(request)

    # The engine runs in a worker thread. With --max-concurrency N, at most N requests run at
    # once and the rest wait on the event loop, holding no thread and no GPU memory.
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    async def systemone(
        request: SystemOneRequest,
        http: Request,
        eng: Any = Depends(get_engine),
    ) -> JSONResponse:
        started = time.perf_counter()
        try:
            response = await anyio.to_thread.run_sync(
                evaluate, eng, request, limiter=http.app.state.limiter
            )
        except ValueError as exc:
            # Malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LlamaServerError as exc:
            # The llama-server behind this one failed: not the caller's error.
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    device_map: str | None = None,
    warmup: bool = True,
    max_concurrency: int | None = None,
    attention: str | None = None,
    revision: str | None = None,
    engine: str = "local",
    llama_server_url: str | None = None,
    llama_server_timeout: float = 60.0,
    noul_floor: float | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
        device_map=device_map,
        warmup=warmup,
        max_concurrency=max_concurrency,
        attention=attention,
        revision=revision,
        engine=engine,
        llama_server_url=llama_server_url,
        llama_server_timeout=llama_server_timeout,
        noul_floor=noul_floor,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
