"""The teacher pipeline, against a fake Ollama server (no model needed): licence gate, quality filter,
graph filter, manifest, early stop, helpful errors, and the shell command end to end."""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import teacher  # noqa: E402
from carbide_modules.graphmem import GraphStore  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOOD = ("The old house stood at the end of the quiet road. Every morning the light came through the "
        "kitchen window and warmed the wooden floor, and a small dog waited by the door for someone to come home.")
MARKDOWN = "**Title**\n\n" + GOOD.replace("'", "’") + "\n\n# more\n"
NON_ASCII = "你好世界 " * 30
REPEAT = "the cat sat on the mat " * 20
SHORT = "too short"
GIBBERISH = "zorp flimble quarnt dwex morvle skint blorf yindle crax vomble glint fritz plonk quibb snarf wex"


class _Fake(BaseHTTPRequestHandler):
    replies = []          # texts to serve, in order
    prompts = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "fake:latest", "digest": "abc123"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.prompts.append(req)
        text = _Fake.replies[(len(_Fake.prompts) - 1) % len(_Fake.replies)]
        self._send({"response": text})


def _server(replies):
    _Fake.replies, _Fake.prompts = list(replies), []
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _quiet(*a, **k):
    pass


def test_it_refuses_to_run_until_the_licence_is_acknowledged():
    srv, url = _server([GOOD])
    out = os.path.join(tempfile.mkdtemp(), "t.txt")
    try:
        teacher.run("fake:latest", "story", 2, out, url=url, log=_quiet)
    except teacher.TeacherError as e:
        assert "licence" in str(e) and "--license-ok" in str(e)
    else:
        raise AssertionError("must refuse without the licence acknowledgement")
    assert not os.path.exists(out) and _Fake.prompts == [], "nothing may be requested or written"
    srv.shutdown()


def test_only_good_text_is_kept_and_every_attempt_is_in_the_manifest():
    srv, url = _server([GOOD, NON_ASCII, REPEAT, SHORT, MARKDOWN])
    out = os.path.join(tempfile.mkdtemp(), "t.txt")
    stats = teacher.run("fake:latest", "story", 5, out, license_ok=True, url=url, topics=["the sea"], log=_quiet)
    srv.shutdown()
    assert stats == {"kept": 2, "rejected": 3, "attempted": 5}, stats
    text = open(out).read()
    assert "**" not in text and "#" not in text and "’" not in text, "markdown and curly quotes are normalised"
    assert text.count("The old house") == 2 and "你" not in text
    rows = [json.loads(line) for line in open(out + ".manifest.jsonl")]
    assert [r["kept"] for r in rows] == [True, False, False, False, True]
    assert {r["reason"] for r in rows} >= {"ok", "not ASCII", "repetitive", "too short"}
    assert rows[0]["model"] == "fake:latest" and rows[0]["digest"] == "abc123" and rows[0]["seed"] == 1000
    assert rows[0]["prompt"].startswith("Write a short story") and len(rows[0]["sha1"]) == 40


def test_the_graph_rejects_text_made_of_words_it_has_never_seen():
    st = GraphStore(os.path.join(tempfile.mkdtemp(), "g.db"))
    for w in set(GOOD.lower().replace(".", "").replace(",", "").split()):
        st.add_node(w)
    assert teacher.check(GOOD, store=st) == (True, "ok")
    ok, reason = teacher.check(GIBBERISH + " " + GIBBERISH, store=st, min_distinct=0.0)
    assert not ok and "graph" in reason
    assert teacher.check(GIBBERISH + " " + GIBBERISH, store=None, min_distinct=0.0)[0], "no graph, no graph filter"


def test_helpful_errors_for_a_missing_server_and_an_unknown_model():
    try:
        teacher.available_models(url="http://127.0.0.1:9")
    except teacher.TeacherError as e:
        assert "ollama serve" in str(e)
    else:
        raise AssertionError("an unreachable server must give a helpful error")
    srv, url = _server([GOOD])
    try:
        teacher.run("nope", "story", 1, os.path.join(tempfile.mkdtemp(), "t.txt"), license_ok=True, url=url, log=_quiet)
    except teacher.TeacherError as e:
        assert "fake:latest" in str(e), "should list what IS installed"
    else:
        raise AssertionError("unknown model must be refused")
    srv.shutdown()


def test_stop_ends_the_run_early_but_keeps_what_was_done():
    srv, url = _server([GOOD])
    out = os.path.join(tempfile.mkdtemp(), "t.txt")
    calls = iter([False, False, True])
    stats = teacher.run("fake:latest", "explain", 10, out, license_ok=True, url=url, should_stop=lambda: next(calls), log=_quiet)
    srv.shutdown()
    assert stats["attempted"] == 2 and open(out).read().count("The old house") == 2


def test_the_shell_command_end_to_end():
    srv, url = _server([GOOD])
    d = tempfile.mkdtemp()
    with open(os.path.join(REPO, "carbide_training_dataset.txt"), "rb") as f:
        open(os.path.join(d, "c.txt"), "wb").write(f.read(40_000))
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""), CARBIDE_OLLAMA_URL=url)
    script = ["teacher fake:latest story 2 out.txt the sea", "teacher fake:latest story 2 out.txt the sea --license-ok",
              "teacher fake:latest bogusgenre 2 out.txt --license-ok"]
    out = subprocess.run([sys.executable, "-m", "carbide_modules.shell", "--data", "c.txt"], cwd=d, env=env,
                         input="\n".join(script) + "\nquit\n", capture_output=True, text=True, timeout=300).stdout
    srv.shutdown()
    assert "licence" in out and '"kept": 2' in out and "usage: teacher" in out, out
    assert open(os.path.join(d, "out.txt")).read().count("The old house") == 2
