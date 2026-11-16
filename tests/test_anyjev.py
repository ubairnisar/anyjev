import asyncio
import base64

import pytest

from anyjev import Judge, LabelNotFoundError, Option, Task, binary, calibrate_threshold, presets
from anyjev.cli import main as cli_main
from anyjev.inputs import build_content, image_url
from anyjev.task import iter_orders

from .fakes import FakeClient

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

TASK = Task(question="Q?", options={"yes": "Yes", "no": "No"}, positive="yes", threshold=0.3)


def biased(base: dict[str, float], first_bonus: float = 0.0):
    """Fixed probabilities per description, plus a bonus for whatever is shown first."""
    def answer(prompt, descs):
        p = dict(base)
        p[descs[0]] = p[descs[0]] + first_bonus
        z = sum(p[d] for d in descs)
        return {d: p[d] / z for d in descs}
    return answer


# ------------------------------------------------------------------ task
def test_options_accept_many_shapes():
    shapes = (
        {"a": "A!", "b": "B!"},
        [("a", "A!"), ("b", "B!")],
        [Option("a", "A!"), {"key": "b", "description": "B!"}],
    )
    for opts in shapes:
        assert Task(question="q", options=opts).keys == ["a", "b"]


def test_task_validation():
    with pytest.raises(ValueError):
        Task(question="q", options={"a": "x"})
    with pytest.raises(ValueError):
        Task(question="q", options={"a": "x", "b": "y"}, positive="c")
    with pytest.raises(ValueError):
        Task(question="q", options={"a": "x", "b": "y"}, threshold=0.4)


def test_render_maps_letters_to_keys():
    prompt, mapping = TASK.render([1, 0])
    assert "A. No" in prompt and "B. Yes" in prompt
    assert mapping == {"A": "no", "B": "yes"}
    assert prompt.rstrip().endswith("Answer:")


def test_iter_orders():
    assert list(iter_orders(3, "cyclic")) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert list(iter_orders(3, "reverse")) == [[0, 1, 2], [2, 1, 0]]
    assert list(iter_orders(3, "none")) == [[0, 1, 2]]


def test_task_yaml_roundtrip(tmp_path):
    p = tmp_path / "t.yaml"
    presets.structured_vs_prose().save(p)
    t = Task.load(p)
    assert t.positive == "structured" and t.threshold == 0.3 and t.keys == ["structured", "prose"]


# ---------------------------------------------------------------- inputs
def test_image_inputs(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(PNG)
    assert image_url(f).startswith("data:image/png;base64,")
    assert image_url(PNG).startswith("data:image/png;base64,")
    assert image_url("https://e.com/a.png") == "https://e.com/a.png"
    parts = build_content("prompt", image=[f, f], text="hello")
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text", "text"]
    with pytest.raises(FileNotFoundError):
        image_url(tmp_path / "missing.png")


# ----------------------------------------------------------------- judge
def test_debiasing_cancels_position_bias():
    client = FakeClient(biased({"Yes": 0.5, "No": 0.5}, first_bonus=0.5))
    r = Judge(client=client).classify(TASK, text="x")
    assert len(client.calls) == 2
    assert r.probs["yes"] == pytest.approx(0.5)
    assert r.order_gap == pytest.approx(1 / 3)
    assert r.uncertain  # gap > 0.3


def test_no_debias_shows_bias():
    client = FakeClient(biased({"Yes": 0.5, "No": 0.5}, first_bonus=0.5))
    r = Judge(client=client, debias="none").classify(TASK, text="x")
    assert len(client.calls) == 1
    assert r.probs["yes"] == pytest.approx(2 / 3)


def test_threshold_favours_positive():
    client = FakeClient(biased({"Yes": 0.35, "No": 0.65}))
    judge = Judge(client=client)
    r = judge.classify(TASK, text="x")
    assert r.label == "yes" and r.is_positive and r.p_positive == pytest.approx(0.35)
    assert judge.classify(TASK, text="x", threshold=0.5).label == "no"


def test_argmax_multiclass_and_request_shape():
    task = presets.sentiment()
    client = FakeClient(biased({"Positive": 0.2, "Neutral or mixed": 0.1, "Negative": 0.7}))
    r = Judge(client=client).classify(task, text="awful")
    assert r.label == "negative" and len(client.calls) == 3
    call = client.calls[0]
    assert call["max_tokens"] == 1 and call["logprobs"] is True and call["temperature"] == 0
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert call["model"] == "fake-model"


def test_token_variants_are_summed():
    client = FakeClient(biased({"Yes": 0.25, "No": 0.25}), extra_tokens={" A": -0.6931})
    r = Judge(client=client, debias="none").classify(TASK, text="x")
    # fake normalises Yes/No to 0.5 each; " A" adds 0.5 more -> A=1.0, B=0.5
    assert r.probs["yes"] == pytest.approx(2 / 3, abs=1e-3)


def test_missing_label_floor_and_raise():
    def answer(prompt, descs):
        return {descs[0]: 0.9}  # second letter never appears
    client = FakeClient(answer, extra_tokens={"the": -5.0})
    r = Judge(client=client, debias="none").classify(TASK, text="x")
    assert r.uncertain and any("outside the top" in s for s in r.reasons)
    with pytest.raises(LabelNotFoundError):
        Judge(client=client, debias="none", on_missing="raise").classify(TASK, text="x")


def test_thinking_model_gives_helpful_error():
    client = FakeClient(lambda p, d: {}, extra_tokens={"<think>": -0.01})
    with pytest.raises(LabelNotFoundError, match="thinking"):
        Judge(client=client).classify(TASK, text="x")


def test_classify_many_keeps_order_and_isolates_errors(tmp_path):
    client = FakeClient(biased({"Yes": 0.9, "No": 0.1}))
    seen = []
    items = [{"text": "a", "id": "a"}, {"image": tmp_path / "missing.png", "id": "bad"}, {"text": "c"}]
    rs = Judge(client=client).classify_many(TASK, items, concurrency=2, on_result=seen.append)
    assert [r.input_id for r in rs] == ["a", "bad", "2"]
    assert rs[1].error and "FileNotFoundError" in rs[1].error and not rs[1].ok
    assert rs[0].label == "yes" and len(seen) == 3


def test_async_api():
    client = FakeClient(biased({"Yes": 0.9, "No": 0.1}))

    async def go():
        async with Judge(client=client) as j:
            one = await j.aclassify(TASK, text="x")
            many = await j.aclassify_many(TASK, [{"text": "y"}, {"text": "z"}])
        return one, many

    one, many = asyncio.run(go())
    assert one.label == "yes" and len(many) == 2


def test_binary_helper():
    t = binary("Is it a cat?", "Yes, a cat", "No", threshold=0.2)
    assert t.positive == "yes" and t.threshold == 0.2


def test_no_input_raises():
    with pytest.raises(ValueError):
        Judge(client=FakeClient(biased({"Yes": 1, "No": 1}))).classify(TASK)


# ------------------------------------------------------------- calibrate
def test_calibrate_threshold():
    scores = [0.9, 0.6, 0.35, 0.2, 0.1, 0.05]
    truth = [True, True, True, False, False, False]
    cal = calibrate_threshold(scores, truth, target_recall=1.0)
    assert cal.threshold == 0.35 and cal.recall == 1.0 and cal.false_positive_rate == 0.0
    cal = calibrate_threshold(scores, truth, target_recall=0.6)
    assert cal.threshold == 0.6
    assert "threshold" in str(cal)


def test_judge_calibrate():
    def answer(prompt, descs):
        return {d: 0.5 for d in descs}
    j = Judge(client=FakeClient(answer))
    cal = j.calibrate(TASK, [{"text": "a"}, {"text": "b"}], ["yes", "no"])
    assert cal.n_positive == 1 and cal.threshold == pytest.approx(0.5)


# ------------------------------------------------------------------- cli
def test_cli_presets_and_init(tmp_path, capsys):
    assert cli_main(["presets"]) == 0
    assert "structured-vs-prose" in capsys.readouterr().out
    target = tmp_path / "mine.yaml"
    assert cli_main(["init", str(target)]) == 0
    t = Task.load(target)
    assert t.name == "mine" and t.positive == "technical"
