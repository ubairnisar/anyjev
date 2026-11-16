"""Task definition: the question, the options, and the decision rule."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Option:
    """One answer the model can pick.

    key:          stable identifier you get back in results (e.g. "structured").
    description:  what the model reads next to the letter.
    """

    key: str
    description: str


OptionsLike = Union[Mapping[str, str], Sequence[Union[Option, tuple, str, Mapping[str, str]]]]


def _normalize_options(options: OptionsLike) -> list[Option]:
    if isinstance(options, Mapping):
        return [Option(str(k), str(v)) for k, v in options.items()]

    out: list[Option] = []
    for o in options:
        if isinstance(o, Option):
            out.append(o)
        elif isinstance(o, tuple) and len(o) == 2:
            out.append(Option(str(o[0]), str(o[1])))
        elif isinstance(o, str):
            out.append(Option(o, o))
        elif isinstance(o, Mapping) and "key" in o:
            out.append(Option(str(o["key"]), str(o.get("description", o["key"]))))
        else:
            raise TypeError(
                f"Cannot interpret option {o!r}. Use Option(key, description), "
                "a (key, description) tuple, a string, or a {key, description} dict."
            )
    return out


@dataclass
class Task:
    """A multiple-choice judgement to run against an image and/or text.

    Example::

        task = Task(
            question="Is this review positive or negative?",
            options={"pos": "Positive", "neg": "Negative"},
        )

    For "when in doubt, say X" tasks, set ``positive`` and a ``threshold``
    lower than 0.5: the result is ``positive`` whenever P(positive) reaches the
    threshold, even if another option scores higher.
    """

    question: str
    options: OptionsLike
    context: str = ""
    name: str = "task"
    positive: str | None = None
    threshold: float | None = None
    system: str | None = None
    answer_instruction: str = "Answer with a single letter."
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.options = _normalize_options(self.options)
        keys = [o.key for o in self.options]
        if len(keys) < 2:
            raise ValueError("A task needs at least 2 options.")
        if len(keys) > len(LETTERS):
            raise ValueError(f"A task supports at most {len(LETTERS)} options.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Option keys must be unique, got {keys}.")
        if self.positive is not None and self.positive not in keys:
            raise ValueError(f"positive={self.positive!r} is not one of the option keys {keys}.")
        if self.threshold is not None:
            if not 0.0 <= self.threshold <= 1.0:
                raise ValueError("threshold must be between 0 and 1.")
            if self.positive is None:
                raise ValueError("threshold needs positive=<option key> to know which option it applies to.")

    # ------------------------------------------------------------------ keys
    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]  # type: ignore[union-attr]

    # ---------------------------------------------------------------- prompt
    def render(self, order: Sequence[int] | None = None) -> tuple[str, dict[str, str]]:
        """Build the prompt for one option order.

        Returns (prompt_text, {letter: option_key}).
        """
        opts: list[Option] = self.options  # type: ignore[assignment]
        order = list(range(len(opts))) if order is None else list(order)
        if sorted(order) != list(range(len(opts))):
            raise ValueError(f"order must be a permutation of 0..{len(opts) - 1}, got {order}")

        letter_to_key: dict[str, str] = {}
        lines = [self.question.strip()]
        if self.context.strip():
            lines += ["", self.context.strip()]
        lines += ["", "Options:"]
        for letter, idx in zip(LETTERS, order, strict=False):
            lines.append(f"{letter}. {opts[idx].description}")
            letter_to_key[letter] = opts[idx].key
        lines += ["", self.answer_instruction, "Answer:"]
        return "\n".join(lines), letter_to_key

    # ------------------------------------------------------------- (de)serde
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "question": self.question,
            "context": self.context,
            "options": [{"key": o.key, "description": o.description} for o in self.options],  # type: ignore[union-attr]
        }
        for k in ("positive", "threshold", "system"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.answer_instruction != Task.answer_instruction:
            d["answer_instruction"] = self.answer_instruction
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        allowed = {"name", "question", "context", "options", "positive", "threshold",
                   "system", "answer_instruction", "metadata"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown task fields: {sorted(unknown)}. Allowed: {sorted(allowed)}")
        return cls(**dict(data))

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        """Load a task from a .yaml/.yml or .json file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        task = cls.from_dict(data)
        if task.name == "task":
            task.name = path.stem
        return task

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def binary(
    question: str,
    yes: str,
    no: str,
    *,
    yes_key: str = "yes",
    no_key: str = "no",
    context: str = "",
    threshold: float | None = None,
    name: str = "task",
) -> Task:
    """Shortcut for a two-option task with an optional bias toward ``yes``."""
    return Task(
        name=name,
        question=question,
        context=context,
        options=[Option(yes_key, yes), Option(no_key, no)],
        positive=yes_key,
        threshold=threshold,
    )


def iter_orders(n: int, debias: str) -> Iterable[list[int]]:
    """Option orders to score. Averaging over them cancels position bias."""
    base = list(range(n))
    if debias == "none":
        return [base]
    if debias == "reverse":
        return [base, base[::-1]]
    if debias == "cyclic":
        return [base[i:] + base[:i] for i in range(n)]
    raise ValueError(f"debias must be 'none', 'reverse' or 'cyclic', got {debias!r}")
