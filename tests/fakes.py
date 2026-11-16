"""A fake AsyncOpenAI client that answers from a scripted probability table."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace as NS
from typing import Callable


class FakeClient:
    """``answer(prompt, option_descriptions_in_order) -> {description: prob}``.

    The fake maps those probabilities back to letters, so tests can express
    position bias ("first option gets +0.2") independently of option order.
    """

    def __init__(self, answer: Callable[[str, list[str]], dict[str, float]],
                 models=("fake-model",), extra_tokens: dict[str, float] | None = None,
                 sampled: str | None = None):
        self.answer = answer
        self.calls: list[dict] = []
        self.extra_tokens = extra_tokens or {}
        self.sampled = sampled
        self.chat = NS(completions=NS(create=self._create))
        self.models = NS(list=self._list)
        self._models = models

    async def _list(self):
        return NS(data=[NS(id=m) for m in self._models])

    async def _create(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][-1]["content"][-1]["text"]
        lines = re.findall(r"^([A-Z])\. (.*)$", prompt, flags=re.M)
        descs = [d for _, d in lines]
        probs = self.answer(prompt, descs)
        tokens = {letter: math.log(probs[d]) for (letter, d) in lines if probs.get(d, 0) > 0}
        tokens.update(self.extra_tokens)
        ranked = sorted(tokens.items(), key=lambda kv: -kv[1])[: kw.get("top_logprobs", 20)]
        top = [NS(token=t, logprob=lp) for t, lp in ranked]
        first_tok = self.sampled or ranked[0][0]
        first = NS(token=first_tok, logprob=tokens.get(first_tok, -50.0), top_logprobs=top)
        return NS(choices=[NS(logprobs=NS(content=[first]))])

    async def close(self):
        pass
