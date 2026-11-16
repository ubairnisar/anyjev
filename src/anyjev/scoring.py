"""Pure math: read label logprobs from a response and turn them into probabilities."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


class LabelNotFoundError(RuntimeError):
    """A label letter was missing from the returned top logprobs."""


def logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    m = max(logits.values())
    e = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def first_token_logprobs(response: Any) -> tuple[str, dict[str, float]]:
    """Return (sampled_token, {token: logprob}) for the first generated position."""
    choice = response.choices[0]
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        raise RuntimeError(
            "The server returned no logprobs. Make sure it supports logprobs=True "
            "(vLLM does; start it with --max-logprobs >= top_logprobs)."
        )
    first = content[0]
    seen: dict[str, float] = {first.token: first.logprob}
    for cand in first.top_logprobs or []:
        seen.setdefault(cand.token, cand.logprob)
    return first.token, seen


def label_logprobs(
    seen: Mapping[str, float],
    letters: Iterable[str],
    *,
    on_missing: str = "floor",
    sampled: str = "",
) -> tuple[dict[str, float], list[str]]:
    """Collect logprobs for each label letter.

    Tokens like "A", " A" and "A\\n" all count for letter "A" (their mass is summed).
    A letter absent from the top-k is either an error (on_missing="raise") or gets
    the lowest returned logprob (on_missing="floor"), which is an upper bound on
    its true value. Missing letters are returned so callers can flag the result.
    """
    letters = list(letters)
    out: dict[str, float] = {}
    missing: list[str] = []
    for letter in letters:
        vals = [v for tok, v in seen.items() if tok.strip() == letter]
        if vals:
            out[letter] = logsumexp(vals)
        else:
            missing.append(letter)

    if missing:
        if on_missing == "raise" or not out:
            hint = (
                "The first generated token is not an answer letter; the model is probably "
                "thinking or chatting first. Keep disable_thinking=True or try constrain=True."
                if sampled.strip() not in letters
                else "Raise top_logprobs (and vLLM's --max-logprobs)."
            )
            raise LabelNotFoundError(
                f"Labels {missing} not in the returned logprobs. Sampled token: {sampled!r}. "
                f"Returned tokens: {list(seen)[:10]}... {hint}"
            )
        floor = min(seen.values())
        for letter in missing:
            out[letter] = floor
    return out, missing
