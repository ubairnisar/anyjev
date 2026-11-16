"""The Judge: runs a Task against any OpenAI-compatible server (vLLM first)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import urllib.request
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from .inputs import ImageLike, build_content
from .result import OrderScore, Result
from .scoring import first_token_logprobs, label_logprobs, softmax
from .task import LETTERS, Task, iter_orders

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _run(coro: Awaitable[Any]) -> Any:
    """Run a coroutine from sync code, even inside Jupyter's running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


class Judge:
    """Scores multiple-choice Tasks with first-token logprobs.

    Quick start::

        from anyjev import Judge, presets

        judge = Judge("http://localhost:8000/v1")          # model auto-detected
        r = judge.classify(presets.structured_vs_prose(), image="crop.png")
        print(r.label, r.probs, r.uncertain)

    Every setting falls back to an environment variable, so the same code
    runs against different servers: ANYJEV_BASE_URL, ANYJEV_MODEL, ANYJEV_API_KEY.

    Args:
        base_url:         OpenAI-compatible endpoint, e.g. vLLM's ``http://host:8000/v1``.
        model:            served model name. ``None`` = first model the server lists.
        api_key:          only needed if the server was started with ``--api-key``.
        debias:           "cyclic" (score every rotation of the options; for 2 options
                          this is both orders), "reverse", or "none".
        top_logprobs:     how many candidates to request for the first token.
        disable_thinking: pass ``enable_thinking=False`` to the chat template so
                          reasoning models (Qwen3, ...) answer with the letter first.
        constrain:        also restrict sampling to the label tokens (vLLM
                          ``allowed_token_ids``); token ids come from vLLM's /tokenize.
        max_order_gap:    mark a result uncertain when option orders disagree by more than this.
        on_missing:       "floor" (default) or "raise" when a label is not in the top-k.
        concurrency:      default parallel requests for batch methods.
        extra_body:       merged into every request (any vLLM sampling extra).
        client:           bring your own ``openai.AsyncOpenAI`` instance.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        *,
        debias: str = "cyclic",
        top_logprobs: int = 20,
        disable_thinking: bool = True,
        constrain: bool = False,
        max_order_gap: float = 0.3,
        on_missing: str = "floor",
        concurrency: int = 8,
        timeout: float = 120.0,
        max_retries: int = 2,
        extra_body: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        iter_orders(2, debias)  # validate early
        if on_missing not in ("floor", "raise"):
            raise ValueError("on_missing must be 'floor' or 'raise'.")
        self.base_url = (base_url or os.environ.get("ANYJEV_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("ANYJEV_MODEL") or None
        self.api_key = api_key or os.environ.get("ANYJEV_API_KEY") or "EMPTY"
        self.debias = debias
        self.top_logprobs = top_logprobs
        self.disable_thinking = disable_thinking
        self.constrain = constrain
        self.max_order_gap = max_order_gap
        self.on_missing = on_missing
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self._user_client = client
        self._aclient: Any = None
        self._token_ids: dict[str, list[int]] = {}

    def __repr__(self) -> str:
        return f"Judge(base_url={self.base_url!r}, model={self.model!r}, debias={self.debias!r})"

    # ------------------------------------------------------------ clients
    def _new_client(self) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key,
                           timeout=self.timeout, max_retries=self.max_retries)

    def _async_client(self) -> Any:
        if self._user_client is not None:
            return self._user_client
        if self._aclient is None:
            self._aclient = self._new_client()
        return self._aclient

    def _sync(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Run an internal coroutine with a client bound to the current event loop."""

        async def runner() -> Any:
            if self._user_client is not None:
                return await fn(self._user_client, *args, **kwargs)
            client = self._new_client()
            try:
                return await fn(client, *args, **kwargs)
            finally:
                await client.close()

        return _run(runner())

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.close()
            self._aclient = None

    async def __aenter__(self) -> "Judge":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------- server
    async def _resolve_model(self, client: Any) -> str:
        if self.model:
            return self.model
        models = await client.models.list()
        data = getattr(models, "data", models)
        if not data:
            raise RuntimeError(f"No models served at {self.base_url}.")
        self.model = data[0].id
        return self.model

    def _server_root(self) -> str:
        root = self.base_url
        return root[: -len("/v1")] if root.endswith("/v1") else root

    async def _tokenize(self, model: str, text: str) -> list[int]:
        """vLLM's /tokenize endpoint (served next to /v1)."""

        def call() -> list[int]:
            req = urllib.request.Request(
                f"{self._server_root()}/tokenize",
                data=json.dumps({"model": model, "prompt": text, "add_special_tokens": False}).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return list(json.loads(resp.read())["tokens"])

        return await asyncio.to_thread(call)

    async def _label_token_ids(self, model: str, n: int) -> list[int]:
        """Single-token ids for "A", " A", ... (used by constrain=True)."""
        ids: list[int] = []
        for letter in LETTERS[:n]:
            if letter not in self._token_ids:
                found = []
                for variant in (letter, " " + letter):
                    toks = await self._tokenize(model, variant)
                    if len(toks) == 1:
                        found.append(toks[0])
                if not found:
                    raise RuntimeError(f"Label {letter!r} is not a single token for {model}.")
                self._token_ids[letter] = found
            ids.extend(self._token_ids[letter])
        return sorted(set(ids))

    # -------------------------------------------------------------- core
    async def _score_order(
        self, client: Any, model: str, task: Task, order: list[int],
        image: Any, text: str | None,
    ) -> OrderScore:
        prompt, letter_to_key = task.render(order)
        messages: list[dict[str, Any]] = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": build_content(prompt, image=image, text=text)})

        body: dict[str, Any] = dict(self.extra_body)
        if self.disable_thinking:
            kw = dict(body.get("chat_template_kwargs", {}))
            kw.setdefault("enable_thinking", False)
            body["chat_template_kwargs"] = kw
        if self.constrain:
            body["allowed_token_ids"] = await self._label_token_ids(model, len(order))

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body=body or None,
        )

        sampled, seen = first_token_logprobs(response)
        letters = list(letter_to_key)
        logits, missing = label_logprobs(seen, letters, on_missing=self.on_missing, sampled=sampled)
        probs = softmax(logits)
        return OrderScore(
            order=[letter_to_key[l] for l in letters],
            probs={letter_to_key[l]: p for l, p in probs.items()},
            sampled_token=sampled,
            missing=[letter_to_key[l] for l in missing],
        )

    async def _classify(
        self, client: Any, task: Task, *, image: Any = None, text: str | None = None,
        threshold: float | None = None, input_id: str | None = None,
    ) -> Result:
        if image is None and text is None:
            raise ValueError("Pass image=..., text=..., or both.")
        model = await self._resolve_model(client)
        orders = list(iter_orders(len(task.keys), self.debias))
        scores = await asyncio.gather(
            *(self._score_order(client, model, task, o, image, text) for o in orders)
        )
        return self._combine(task, scores, threshold=threshold, input_id=input_id)

    def _combine(
        self, task: Task, scores: Sequence[OrderScore], *,
        threshold: float | None = None, input_id: str | None = None,
    ) -> Result:
        keys = task.keys
        probs = {k: sum(s.probs[k] for s in scores) / len(scores) for k in keys}
        gap = max(max(s.probs[k] for s in scores) - min(s.probs[k] for s in scores) for k in keys)

        thr = task.threshold if threshold is None else threshold
        if task.positive is not None and thr is not None:
            if probs[task.positive] >= thr:
                label = task.positive
            else:
                label = max((k for k in keys if k != task.positive), key=probs.__getitem__)
        else:
            label = max(keys, key=probs.__getitem__)

        reasons: list[str] = []
        if gap > self.max_order_gap:
            reasons.append(f"option orders disagree (gap {gap:.2f} > {self.max_order_gap})")
        missing = sorted({m for s in scores for m in s.missing})
        if missing:
            reasons.append(f"labels {missing} were outside the top-{self.top_logprobs} logprobs")

        return Result(
            task=task.name, label=label, probs=probs, confidence=probs[label],
            order_gap=gap, uncertain=bool(reasons), reasons=reasons,
            positive=task.positive, threshold=thr, per_order=list(scores), input_id=input_id,
        )

    async def _classify_many(
        self, client: Any, task: Task, items: Iterable[Any], *,
        concurrency: int | None = None, threshold: float | None = None,
        on_result: Callable[[Result], None] | None = None, raise_errors: bool = False,
    ) -> list[Result]:
        sem = asyncio.Semaphore(concurrency or self.concurrency)
        await self._resolve_model(client)

        async def one(i: int, item: Any) -> Result:
            image, text, item_id = _unpack(item, i)
            async with sem:
                try:
                    r = await self._classify(client, task, image=image, text=text,
                                             threshold=threshold, input_id=item_id)
                except Exception as e:  # noqa: BLE001 - one bad item must not sink a batch
                    if raise_errors:
                        raise
                    r = Result.failed(task.name, e, input_id=item_id)
            if on_result is not None:
                on_result(r)
            return r

        return list(await asyncio.gather(*(one(i, it) for i, it in enumerate(items))))

    # -------------------------------------------------------- public API
    async def aclassify(
        self, task: Task, *, image: ImageLike | Sequence[ImageLike] | None = None,
        text: str | None = None, threshold: float | None = None,
    ) -> Result:
        """Score one input (async)."""
        return await self._classify(self._async_client(), task, image=image, text=text, threshold=threshold)

    def classify(
        self, task: Task, *, image: ImageLike | Sequence[ImageLike] | None = None,
        text: str | None = None, threshold: float | None = None,
    ) -> Result:
        """Score one input.

        ``image`` may be a path, URL, bytes, PIL image, or a list of them.
        ``text`` is shown to the model as the item to judge.
        ``threshold`` overrides the task's threshold for this call.
        """
        return self._sync(self._classify, task, image=image, text=text, threshold=threshold)

    __call__ = classify

    async def aclassify_many(self, task: Task, items: Iterable[Any], **kwargs: Any) -> list[Result]:
        """Score many inputs concurrently (async). See ``classify_many``."""
        return await self._classify_many(self._async_client(), task, items, **kwargs)

    def classify_many(
        self, task: Task, items: Iterable[Any], *, concurrency: int | None = None,
        threshold: float | None = None, on_result: Callable[[Result], None] | None = None,
        raise_errors: bool = False,
    ) -> list[Result]:
        """Score many inputs concurrently; results keep input order.

        Each item is an image (path/URL/bytes/PIL) or a dict with any of
        ``image``, ``text``, ``id``. Failed items come back as results with
        ``.error`` set unless ``raise_errors=True``.
        """
        return self._sync(self._classify_many, task, list(items), concurrency=concurrency,
                          threshold=threshold, on_result=on_result, raise_errors=raise_errors)

    def calibrate(
        self, task: Task, items: Iterable[Any], labels: Iterable[Any], *,
        target_recall: float = 1.0, concurrency: int | None = None,
    ):
        """Score labelled items and pick the threshold for ``task.positive``.

        ``labels`` are booleans, or option keys (compared against ``task.positive``).
        Returns a :class:`~anyjev.calibrate.Calibration`; apply it with
        ``task.threshold = cal.threshold``.
        """
        from .calibrate import calibrate_threshold

        if task.positive is None:
            raise ValueError("calibrate() needs a task with positive=<option key>.")
        items = list(items)
        truth = [l if isinstance(l, bool) else str(l) == task.positive for l in labels]
        if len(truth) != len(items):
            raise ValueError("items and labels must have the same length.")
        results = self.classify_many(task, items, concurrency=concurrency, raise_errors=True)
        return calibrate_threshold([r.p_positive for r in results], truth, target_recall=target_recall)

    # ------------------------------------------------------------- doctor
    def doctor(self) -> list[tuple[str, bool, str]]:
        """Check the server end to end. Returns [(check, passed, detail)]."""
        return self._sync(self._doctor)

    async def _doctor(self, client: Any) -> list[tuple[str, bool, str]]:
        checks: list[tuple[str, bool, str]] = []
        try:
            models = await client.models.list()
            names = [m.id for m in getattr(models, "data", models)]
            checks.append(("server reachable", True, f"{self.base_url} serves {names}"))
        except Exception as e:  # noqa: BLE001
            checks.append(("server reachable", False, f"{self.base_url}: {e}"))
            return checks

        model = await self._resolve_model(client)
        if names and model not in names:
            checks.append(("model served", False, f"{model!r} not in {names}"))
            return checks
        checks.append(("model served", True, model))

        try:
            ids = {l: await self._tokenize(model, l) for l in "AB"}
            single = all(len(v) == 1 for v in ids.values())
            checks.append(("labels are single tokens", single, str(ids)))
        except Exception as e:  # noqa: BLE001
            checks.append(("labels are single tokens", True, f"skipped (/tokenize unavailable: {e})"))

        probe = Task(question="Which letter comes first in the alphabet?",
                     options={"a": "The letter A", "b": "The letter Z"}, name="doctor")
        try:
            score = await self._score_order(client, model, probe, [0, 1], None, None)
            letter_first = score.sampled_token.strip() in ("A", "B")
            detail = f"first token {score.sampled_token!r}, probs {score.probs}"
            if not letter_first:
                detail += " - model is not answering with a letter first (thinking enabled?)"
            checks.append(("logprobs + letter answer", letter_first, detail))
        except Exception as e:  # noqa: BLE001
            checks.append(("logprobs + letter answer", False, str(e)))
        return checks


def _unpack(item: Any, index: int) -> tuple[Any, str | None, str]:
    if isinstance(item, Mapping):
        image, text = item.get("image"), item.get("text")
        item_id = item.get("id")
        if item_id is None:
            item_id = str(image) if isinstance(image, str) else str(index)
        return image, text, str(item_id)
    item_id = str(item) if isinstance(item, (str, os.PathLike)) else str(index)
    return item, None, item_id
