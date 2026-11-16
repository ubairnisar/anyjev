# anyjev

**Run JEV judgements with any LLM you can serve with vLLM.**

anyjev turns a model into a calibrated classifier. Instead of parsing what the
model *writes*, it asks a multiple-choice question, reads the **probability of
every option** from the first-token logprobs, cancels the model's
**position bias**, and makes the call with a **threshold you calibrate**.

```python
from anyjev import Judge, presets

judge = Judge("http://localhost:8000/v1")
result = judge.classify(presets.structured_vs_prose(), image="crop.png")

result.label        # 'structured'
result.probs        # {'structured': 0.41, 'prose': 0.59}
result.uncertain    # False
```

`structured` wins here even though `prose` scores higher. The task says "when
in doubt, choose structured" and sets a threshold of 0.30, so 0.41 is enough.

- **Any model:** text or vision, any model vLLM can serve (Qwen, Llama,
  Gemma, Mistral, InternVL and others). It also works with other
  OpenAI-compatible servers that return logprobs.
- **Probabilities, not strings:** one token per request. Nothing to parse, and
  no "Sure! The answer is…".
- **Position bias removed:** every option order is scored and averaged. A
  large disagreement between orders marks the result `uncertain`.
- **Asymmetric decisions:** for "favour X when in doubt" tasks, set a
  threshold below 0.5 and pick it from labelled data in one command.
- **Ready for real batches:** sync and async APIs, a concurrency limit,
  per-item error isolation, and a CLI.

---

## Contents

- [How it works](#how-it-works)
- [Install](#install)
- [1. Serve a model with vLLM](#1-serve-a-model-with-vllm)
- [2. Check the setup](#2-check-the-setup)
- [3. Judge something](#3-judge-something)
- [Defining tasks](#defining-tasks)
- [Inputs](#inputs)
- [Reading results](#reading-results)
- [Thresholds and calibration](#thresholds-and-calibration)
- [Batches and async](#batches-and-async)
- [CLI reference](#cli-reference)
- [Judge options](#judge-options)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)

---

## How it works

```
             ┌───────────────────── one request per option order ─────────────────────┐
image/text ─▶│ question + context + "A. option 1 / B. option 2 / …" + "Answer:"         │─▶ vLLM
             └──────────────────────────────────────────────────────────────────────────┘
                                                   │  max_tokens=1, logprobs, top-k=20
                                                   ▼
                         P(A), P(B), …  read from the first token (" A" and "A" summed)
                                                   │  softmax over the labels only
                                                   ▼
                         map letters back to option keys, average over all orders
                                                   │
                                                   ▼
                  label = positive if P(positive) ≥ threshold, else argmax
                  uncertain = orders disagree too much, or a label fell outside the top-k
```

1. **Letters, not words.** Options are shown as `A.`, `B.`, `C.` … so each
   answer is a single token and its probability is exact.
2. **Thinking switched off.** Reasoning models (Qwen3 and others) open with
   `<think>`. anyjev passes `enable_thinking=False` to the chat template so
   the first token is the answer.
3. **Debiasing.** Models prefer whichever option comes first, or last.
   anyjev scores every rotation of the options and averages them. For two
   options that means both orders. The requests run concurrently.
4. **Decision.** Use plain argmax, or a "positive" option with its own
   threshold for tasks where a miss costs more than a false alarm.

## Install

```bash
pip install anyjev            # once it is published to PyPI
# or from source:
pip install git+https://github.com/ubairnisar/anyjev
```

Requires Python 3.9 or later. The only runtime dependencies are `openai` and `pyyaml`.
Install `anyjev[pil]` to pass PIL images.

## 1. Serve a model with vLLM

Any model works. To judge images, use a vision-language model:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-logprobs 20
```

```bash
# FP8 checkpoint with a custom served name
vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --served-model-name qwen3-vl --max-logprobs 20
```

```bash
# text-only judging
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-logprobs 20
```

`--max-logprobs` must be at least the `top_logprobs` anyjev requests (20 by default).

## 2. Check the setup

```bash
anyjev doctor --base-url http://gpu-box:8000/v1
```

```
[ok] server reachable: http://gpu-box:8000/v1 serves ['qwen3-vl']
[ok] model served: qwen3-vl
[ok] labels are single tokens: {'A': [32], 'B': [33]}
[ok] logprobs + letter answer: first token 'A', probs {'a': 0.998, 'b': 0.002}
```

To avoid repeating flags, set environment variables. Every API call and
command reads them:

```bash
export ANYJEV_BASE_URL=http://gpu-box:8000/v1
export ANYJEV_MODEL=qwen3-vl        # optional; default is the first served model
export ANYJEV_API_KEY=...           # only if vLLM runs with --api-key
```

## 3. Judge something

**Python**

```python
from anyjev import Judge, presets

judge = Judge()                                   # reads ANYJEV_* or localhost:8000
task = presets.structured_vs_prose()

r = judge.classify(task, image="crop.png")
print(r)
# <Result structured (structured=0.412, prose=0.588) gap=0.031>
```

**CLI**

```bash
anyjev run -t structured-vs-prose crops/
```

```
structured P(structured)=0.912   gap=0.020  crops/001.png
prose      P(structured)=0.043   gap=0.011  crops/002.png
structured P(structured)=0.412   gap=0.031  crops/003.png
structured P(structured)=0.338   gap=0.354  ?  crops/004.png
# structured: 3, prose: 1; uncertain: 1; failed: 0
```

`?` marks an uncertain result. Add `--json` for one JSON object per line,
ready for `jq` or a pipeline.

## Defining tasks

A task is a question, some options, and optionally a decision rule.

**In Python**

```python
from anyjev import Task

router = Task(
    name="ticket-router",
    question="Which team should handle this support ticket?",
    context="Tickets about invoices always go to billing.",   # definitions, rules, examples
    options={
        "billing": "Billing - payments, invoices, refunds",
        "bug":     "Engineering - something is broken",
        "account": "Accounts - login, password, profile",
        "other":   "Anything else",
    },
)

judge.classify(router, text="I was charged twice this month").label   # 'billing'
```

The option **key** (`billing`) is what you get back. The **description** is
what the model reads. You can pass options as a dict, as `(key, description)`
tuples, or as `Option(key, description)` objects.

**In YAML** (share tasks between people and code without touching Python):

```yaml
# tasks/answer-grader.yaml
name: answer-grader
system: You are a strict grader.
question: |
  The input has a question, a reference answer and a student answer.
  Grade the student answer.
options:
  - key: full
    description: Fully correct
  - key: partial
    description: Partially correct - right idea, missing or wrong details
  - key: wrong
    description: Wrong or off-topic
```

```python
task = Task.load("tasks/answer-grader.yaml")
```

```bash
anyjev run -t tasks/answer-grader.yaml answers/*.txt
```

Generate a commented starter file with `anyjev init my-task.yaml`, or copy a
preset with `anyjev init my-task.yaml --from-preset structured-vs-prose`.

**Yes/no shortcut**

```python
from anyjev import binary

is_invoice = binary("Is this page an invoice?", yes="Yes, an invoice", no="No", threshold=0.4)
```

**Built-in presets** (`anyjev presets` lists them, `anyjev presets NAME` prints one as YAML)

| preset                | input | options                                  | rule                       |
|-----------------------|-------|------------------------------------------|----------------------------|
| `structured-vs-prose` | image | structured / prose                       | structured if P ≥ 0.30     |
| `sentiment`           | text  | positive / neutral / negative            | argmax                     |
| `answer-correct`      | text  | correct / incorrect                      | argmax, or threshold       |

### Writing good options

- Put the full definitions in `context`, and keep each option description short and distinct.
- Repeat the option name at the start of its description (`"STRUCTURED - …"`)
  so the model can tie the letter to the definitions.
- Don't add "Answer A or B" yourself. anyjev appends the answer instruction.
- Avoid catch-all options that overlap others, unless you mean "other".

## Inputs

| you pass                                   | the model sees                    |
|--------------------------------------------|-----------------------------------|
| `image="crop.png"` / `Path(...)`           | the image (inlined as base64)     |
| `image="https://…/x.png"` / a `data:` URL  | the URL, fetched by vLLM          |
| `image=png_bytes` / `image=pil_image`      | the image                         |
| `image=[a, b]`                             | several images, in order          |
| `text="..."`                               | the text inside `<input>` tags    |
| both                                       | images, then text, then the task  |

## Reading results

```python
r = judge.classify(task, image="crop.png")

r.label         # decided option key (threshold-aware)
r.probs         # {key: probability}, averaged over orders, sums to 1
r.confidence    # probs[label]
r.order_gap     # max disagreement between option orders (0 = fully consistent)
r.uncertain     # True when the result should be double-checked
r.reasons       # why it is uncertain, e.g. ['option orders disagree (gap 0.35 > 0.3)']
r.per_order     # raw scores per order, including the sampled token
r.is_positive   # tasks with a positive option: label == positive
r.p_positive    # tasks with a positive option: probs[positive]
r.to_dict()     # JSON-ready
```

**Treat `uncertain` as useful information.** A large order gap means the
model's answer depends on the option order, not on the input. Route those
items to a stronger model, a human, or the safe default.

## Thresholds and calibration

Plain argmax means using a threshold of 0.5. When missing a positive costs
more than a false alarm (for example "never skip extraction on a structured
crop"), lower the threshold. Choose it from data instead of guessing.

1. Label 50–200 representative items in a CSV:

   ```csv
   path,label
   crops/001.png,structured
   crops/002.png,prose
   ```

2. Calibrate:

   ```bash
   anyjev calibrate -t structured-vs-prose labels.csv --target-recall 1.0
   ```

   ```
   threshold           0.2714
   recall              1.000   (target 1.000)
   precision           0.874
   false positive rate 0.112
   samples             83 positive / 117 negative

   threshold  recall  precision  fpr
       0.050   1.000      0.611  0.453
       0.100   1.000      0.742  0.248
       ...
   ```

   anyjev picks the **highest** threshold that still meets the target recall,
   which keeps false positives as low as possible. Add `--write` to save it
   into a task file.

3. In Python:

   ```python
   cal = judge.calibrate(task, items=paths, labels=labels, target_recall=0.99)
   task.threshold = cal.threshold
   ```

Calibrate again whenever you change the model, the prompt or the image preprocessing.

## Batches and async

```python
results = judge.classify_many(
    task,
    ["a.png", "b.png", {"image": "c.png", "id": "doc-7/p3"}, {"text": "…", "id": "t1"}],
    concurrency=32,
    on_result=lambda r: print(r),   # streaming progress
)
failed = [r for r in results if not r.ok]   # one bad file never sinks the batch
```

Results keep the order of the inputs. Pass `raise_errors=True` to fail fast instead.

From async code (FastAPI, workers, notebooks with `await`):

```python
async with Judge(concurrency=64) as judge:
    r = await judge.aclassify(task, image="x.png")
    rs = await judge.aclassify_many(task, paths)
```

The sync methods also work inside Jupyter.

## CLI reference

```
anyjev run        -t TASK [inputs…] [--text T]… [--threshold X] [--json [-v]]
anyjev calibrate  -t TASK labels.csv [--target-recall R] [--write]
anyjev doctor
anyjev presets    [NAME]
anyjev init       [PATH] [--from-preset NAME] [--force]

server flags (run, calibrate, doctor):
  --base-url URL  --model NAME  --api-key KEY
  --debias {cyclic,reverse,none}  --top-logprobs N  --concurrency N
  --thinking      leave the chat template's thinking mode on
  --constrain     restrict sampling to label tokens
```

`TASK` is a preset name or a `.yaml`/`.json` file. Inputs can be image files,
`.txt`/`.md` files, directories (searched recursively) or image URLs.
`python -m anyjev` works too.

## Judge options

| option             | default                     | what it does                                                       |
|--------------------|-----------------------------|--------------------------------------------------------------------|
| `base_url`         | `ANYJEV_BASE_URL` / `http://localhost:8000/v1` | the OpenAI-compatible endpoint                   |
| `model`            | `ANYJEV_MODEL` / first served model | the served model name                                      |
| `api_key`          | `ANYJEV_API_KEY` / `"EMPTY"` | needed if vLLM runs with `--api-key`                              |
| `debias`           | `"cyclic"`                  | `cyclic` = every rotation (N requests), `reverse` = 2, `none` = 1  |
| `top_logprobs`     | `20`                        | top-k returned for the first token                                 |
| `disable_thinking` | `True`                      | sends `chat_template_kwargs={"enable_thinking": False}`           |
| `constrain`        | `False`                     | restricts sampling to label tokens with vLLM `allowed_token_ids`   |
| `max_order_gap`    | `0.3`                       | gap above which a result is `uncertain`                            |
| `on_missing`       | `"floor"`                   | a label outside the top-k gets the lowest seen logprob; `"raise"` errors instead |
| `concurrency`      | `8`                         | default parallel requests in batch methods                         |
| `timeout`, `max_retries` | `120`, `2`            | passed to the OpenAI client                                        |
| `extra_body`       | `{}`                        | merged into every request (any vLLM extra parameter)               |
| `client`           | `None`                      | bring your own `openai.AsyncOpenAI`                                |

## Troubleshooting

**`Labels ['B'] not in the returned logprobs. Sampled token: '<think>'`**
The model is reasoning before it answers. Keep `disable_thinking=True` (the
default). If the chat template ignores that flag, add `constrain=True` or use
an instruct (non-thinking) checkpoint.

**`Labels [...] not in the returned logprobs. Sampled token: 'A'`**
The model is so confident that the other letters fell outside the top-k. The
default `on_missing="floor"` handles this and flags the result. You can also
raise `top_logprobs` together with vLLM's `--max-logprobs`.

**`The server returned no logprobs`**
The server does not support `logprobs`. Use vLLM or another server that does.

**Results flip when you reorder options**
That is the position bias anyjev measures. Keep `debias="cyclic"`, and
improve the option descriptions until `order_gap` drops.

**Images are slow or run out of memory**
Downscale crops before sending them. vLLM's `--limit-mm-per-prompt` and the
model's processor settings, such as `max_pixels` for Qwen-VL, control the cost.

## FAQ

**Why not just ask the model and parse the reply?**
A reply gives you one label with no confidence. Logprobs give the full
distribution in a single token, which you can threshold, calibrate, and use
to rank or route items. They are also deterministic and cheaper.

**How many requests does one item cost?**
One per option order: 2 for a two-option task with `cyclic` or `reverse`,
and N for N options with `cyclic`. Each request decodes one token, so the
cost is almost entirely prefill, and vLLM's prefix caching helps with
repeated images. Use `debias="reverse"` to cap it at 2, or `"none"` for 1.

**Does it work with OpenAI, Together, Ollama…?**
It works with any server that implements chat completions with `logprobs`
and `top_logprobs`. `constrain` and the label-token check in `doctor` need
vLLM's `/tokenize`.

**Can I use my own prompt format?**
Yes: `question`, `context`, `system` and `answer_instruction` are all task
fields. The letter list is the one fixed part, because the method depends on it.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
The test suite needs no GPU and runs in seconds:

```bash
pip install -e ".[dev]" && pytest -q
```

## License

[Apache 2.0](LICENSE)
