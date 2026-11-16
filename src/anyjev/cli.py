"""Command line interface: `anyjev run | calibrate | doctor | presets | init`."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, presets
from .inputs import is_image_path
from .judge import Judge
from .result import Result
from .task import Task

TEXT_SUFFIXES = {".txt", ".md"}

INIT_TEMPLATE = """\
# anyjev task. Run it with:  anyjev run -t {path} <images or .txt files>
name: {name}
question: Is the input about a technical topic?
context: |
  Technical topics include programming, engineering, science and maths.
options:
  - key: technical
    description: Technical - the input is mainly about a technical topic
  - key: other
    description: Not technical
# Optional "when in doubt, pick X" rule: choose `positive` when its
# probability reaches `threshold`, even if another option scores higher.
# Find a good threshold with:  anyjev calibrate -t {path} labels.csv
positive: technical
threshold: 0.5
"""


def _load_task(spec: str) -> Task:
    if spec in presets.PRESETS:
        return presets.get(spec)
    path = Path(spec)
    if path.exists():
        return Task.load(path)
    raise SystemExit(f"error: {spec!r} is neither a preset ({', '.join(presets.PRESETS)}) nor a task file.")


def _expand_inputs(paths: list[str], texts: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in paths:
        if p.startswith(("http://", "https://")):
            items.append({"image": p, "id": p})
            continue
        path = Path(p)
        if path.is_dir():
            files = sorted(
                f for f in path.rglob("*")
                if f.is_file() and (is_image_path(f) or f.suffix.lower() in TEXT_SUFFIXES)
            )
        elif path.exists():
            files = [path]
        else:
            raise SystemExit(f"error: no such file: {p}")
        for f in files:
            if f.suffix.lower() in TEXT_SUFFIXES:
                items.append({"text": f.read_text(encoding="utf-8"), "id": str(f)})
            else:
                items.append({"image": str(f), "id": str(f)})
    for i, t in enumerate(texts):
        items.append({"text": t, "id": f"text[{i}]"})
    if not items:
        raise SystemExit("error: nothing to judge. Pass image/text files, a directory, a URL, or --text.")
    return items


def _judge(args: argparse.Namespace) -> Judge:
    return Judge(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        debias=args.debias, top_logprobs=args.top_logprobs,
        disable_thinking=not args.thinking, constrain=args.constrain,
        concurrency=args.concurrency,
    )


def _print_row(r: Result, task: Task) -> None:
    if r.error:
        print(f"ERROR      {r.input_id}  {r.error}", flush=True)
        return
    p = f"P({r.positive})={r.p_positive:.3f}" if r.positive else f"p={r.confidence:.3f}"
    flag = "  ?" if r.uncertain else ""
    print(f"{r.label:<10} {p:<22} gap={r.order_gap:.3f}{flag}  {r.input_id}", flush=True)


# ------------------------------------------------------------------ commands
def cmd_run(args: argparse.Namespace) -> int:
    task = _load_task(args.task)
    items = _expand_inputs(args.inputs, args.text)
    judge = _judge(args)

    if args.json:
        cb = lambda r: print(json.dumps(r.to_dict() if args.verbose else _slim(r)), flush=True)  # noqa: E731
    else:
        cb = lambda r: _print_row(r, task)  # noqa: E731
        print(f"# task={task.name}  model={judge.model or '(auto)'}  items={len(items)}", file=sys.stderr)

    results = judge.classify_many(task, items, threshold=args.threshold, on_result=cb)
    failed = sum(1 for r in results if r.error)
    if not args.json:
        counts: dict[str, int] = {}
        for r in results:
            if r.ok:
                counts[r.label] = counts.get(r.label, 0) + 1
        unc = sum(1 for r in results if r.ok and r.uncertain)
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"# {summary}; uncertain: {unc}; failed: {failed}", file=sys.stderr)
    return 1 if failed == len(results) else 0


def _slim(r: Result) -> dict[str, Any]:
    d = {"id": r.input_id, "label": r.label, "probs": r.probs,
         "order_gap": r.order_gap, "uncertain": r.uncertain}
    if r.reasons:
        d["reasons"] = r.reasons
    if r.error:
        d["error"] = r.error
    return d


def cmd_calibrate(args: argparse.Namespace) -> int:
    task = _load_task(args.task)
    if task.positive is None:
        raise SystemExit("error: calibration needs a task with `positive` set.")
    base = Path(args.labels).parent
    items, truth = [], []
    with open(args.labels, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0].strip().lower() in ("path", "file", "image"):
                continue
            path, label = row[0].strip(), row[1].strip()
            full = path if path.startswith(("http://", "https://")) or Path(path).is_absolute() else str(base / path)
            items.append({"text": Path(full).read_text(encoding="utf-8"), "id": path}
                         if Path(full).suffix.lower() in TEXT_SUFFIXES else {"image": full, "id": path})
            truth.append(label == task.positive or label.lower() in ("1", "true", "yes", "y"))
    if not items:
        raise SystemExit("error: the CSV has no rows. Expected: path,label")

    judge = _judge(args)
    print(f"# scoring {len(items)} labelled items ({sum(truth)} {task.positive})...", file=sys.stderr)
    cal = judge.calibrate(task, items, truth, target_recall=args.target_recall)
    print(cal)
    if args.write:
        task.threshold = round(cal.threshold, 4)
        out = Path(args.task)
        if not out.exists():
            out = Path(f"{task.name}.yaml")
        task.save(out)
        print(f"\nwrote threshold {task.threshold} to {out}")
    else:
        print(f"\nApply it: Task(..., threshold={cal.threshold:.4f})  or rerun with --write")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    for name, passed, detail in _judge(args).doctor():
        ok &= passed
        print(f"[{'ok' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if ok else 1


def cmd_presets(args: argparse.Namespace) -> int:
    if args.name:
        import yaml

        print(yaml.safe_dump(presets.get(args.name).to_dict(), sort_keys=False, allow_unicode=True), end="")
        return 0
    for name, fn in presets.PRESETS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<22} {doc}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        raise SystemExit(f"error: {path} exists (use --force to overwrite).")
    if args.from_preset:
        presets.get(args.from_preset).save(path)
    else:
        path.write_text(INIT_TEMPLATE.format(path=path, name=path.stem), encoding="utf-8")
    print(f"created {path}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anyjev", description="Logprob-scored LLM judgements on any vLLM model.")
    p.add_argument("--version", action="version", version=f"anyjev {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    server = argparse.ArgumentParser(add_help=False)
    g = server.add_argument_group("server")
    g.add_argument("--base-url", help="OpenAI-compatible URL (env ANYJEV_BASE_URL, default http://localhost:8000/v1)")
    g.add_argument("--model", help="served model name (env ANYJEV_MODEL, default: first served model)")
    g.add_argument("--api-key", help="API key (env ANYJEV_API_KEY)")
    g.add_argument("--debias", choices=["cyclic", "reverse", "none"], default="cyclic", help="option-order debiasing")
    g.add_argument("--top-logprobs", type=int, default=20)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--thinking", action="store_true", help="leave the chat template's thinking mode on")
    g.add_argument("--constrain", action="store_true", help="restrict sampling to label tokens (vLLM)")

    r = sub.add_parser("run", parents=[server], help="judge images and/or text")
    r.add_argument("-t", "--task", required=True, help="preset name or task .yaml/.json")
    r.add_argument("inputs", nargs="*", help="image files, .txt/.md files, directories or image URLs")
    r.add_argument("--text", action="append", default=[], help="inline text to judge (repeatable)")
    r.add_argument("--threshold", type=float, help="override the task threshold")
    r.add_argument("--json", action="store_true", help="one JSON object per line")
    r.add_argument("-v", "--verbose", action="store_true", help="with --json: include per-order scores")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("calibrate", parents=[server], help="pick a threshold from a labelled CSV")
    c.add_argument("-t", "--task", required=True)
    c.add_argument("labels", help="CSV rows of: path,label  (label = option key or true/false)")
    c.add_argument("--target-recall", type=float, default=1.0)
    c.add_argument("--write", action="store_true", help="save the threshold into the task file")
    c.set_defaults(fn=cmd_calibrate)

    d = sub.add_parser("doctor", parents=[server], help="check the server, model and logprobs")
    d.set_defaults(fn=cmd_doctor)

    ps = sub.add_parser("presets", help="list built-in tasks, or print one as YAML")
    ps.add_argument("name", nargs="?")
    ps.set_defaults(fn=cmd_presets)

    i = sub.add_parser("init", help="create a starter task file")
    i.add_argument("path", nargs="?", default="task.yaml")
    i.add_argument("--from-preset", metavar="NAME")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
