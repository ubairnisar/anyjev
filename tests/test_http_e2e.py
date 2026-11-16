"""End to end over real HTTP: the openai SDK talks to a tiny vLLM-shaped server."""

import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anyjev import Judge, Task
from anyjev.cli import main as cli_main

TOKENS = {"A": 32, " A": 362, "B": 33, " B": 425}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            model = {"id": "mock-vlm", "object": "model", "created": 0, "owned_by": "x"}
            return self._send({"object": "list", "data": [model]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, req))
        if self.path == "/tokenize":
            return self._send({"tokens": [TOKENS.get(req["prompt"], 0)] if req["prompt"] in TOKENS else [1, 2]})
        prompt = req["messages"][-1]["content"][-1]["text"]
        lines = dict(re.findall(r"^([A-Z])\. (.*)$", prompt, flags=re.M))
        # The "model" says yes with p=0.8 when shown "Yes"/"The letter A".
        good = {l for l, d in lines.items() if d in ("Yes", "The letter A")}
        probs = {l: (0.8 if l in good else 0.2) for l in lines}
        top = [{"token": l, "logprob": math.log(p), "bytes": None} for l, p in probs.items()]
        top.sort(key=lambda t: -t["logprob"])
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
            "choices": [{
                "index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": top[0]["token"]},
                "logprobs": {"content": [{**top[0], "top_logprobs": top}]},
            }],
        })


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


TASK = Task(question="Q?", options={"yes": "Yes", "no": "No"}, positive="yes", threshold=0.5)


def test_sync_classify_over_http(server):
    srv, url = server
    judge = Judge(url, constrain=True)
    r = judge.classify(TASK, text="hello")
    assert judge.model == "mock-vlm"
    assert r.label == "yes" and r.probs["yes"] == pytest.approx(0.8) and not r.uncertain
    chat = [b for p, b in srv.requests if p == "/v1/chat/completions"]
    assert chat[0]["allowed_token_ids"] == sorted(TOKENS.values())
    assert chat[0]["chat_template_kwargs"] == {"enable_thinking": False}
    # the sync API is reusable (fresh client per call)
    assert judge.classify(TASK, text="again").label == "yes"


def test_doctor_over_http(server):
    _, url = server
    checks = Judge(url).doctor()
    assert all(ok for _, ok, _ in checks), checks


def test_cli_run_json(server, tmp_path, capsys):
    _, url = server
    task_file = tmp_path / "t.json"
    TASK.save(task_file)
    (tmp_path / "a.txt").write_text("some text")
    rc = cli_main(["run", "--base-url", url, "-t", str(task_file), str(tmp_path / "a.txt"), "--text", "hi", "--json"])
    assert rc == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [r["label"] for r in rows] == ["yes", "yes"]


def test_cli_calibrate_write(server, tmp_path, capsys):
    _, url = server
    task_file = tmp_path / "t.yaml"
    TASK.save(task_file)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "labels.csv").write_text("path,label\na.txt,yes\na.txt,no\n")
    argv = ["calibrate", "--base-url", url, "-t", str(task_file), str(tmp_path / "labels.csv"), "--write"]
    assert cli_main(argv) == 0
    assert Task.load(task_file).threshold == pytest.approx(0.8)
