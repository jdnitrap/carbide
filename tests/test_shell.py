"""The command shell, driven the way a person (or the TUI) drives it: lines in, text out."""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(REPO, "carbide_training_dataset.txt")


def _session(lines, cwd, extra_env=None):
    """Run `python -m carbide_modules.shell` in `cwd` (so checkpoints/settings land there)."""
    with open(CORPUS, "rb") as f, open(os.path.join(cwd, "corpus.txt"), "wb") as out:
        out.write(f.read(80_000))
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-m", "carbide_modules.shell", "--data", "corpus.txt"], cwd=cwd, env=env,
                       input="\n".join(lines) + "\nquit\n", capture_output=True, text=True, timeout=600)
    return r.stdout


def _small():
    return ["set model.d_model 16", "set model.n_layers 1", "set model.d_state 8", "set train.seq_len 32",
            "set train.batch_size 4"]


def test_status_train_save_generate_and_a_restart_that_remembers():
    d = tempfile.mkdtemp()
    out = _session(_small() + ["status", "train 30", "status", "gen the", "save keep"], d)
    assert "d_model=16" in out and "step=30" in out and "Checkpoint saved" in out
    assert os.path.exists(os.path.join(d, "carbide_checkpoints", "carbide_ckpt.pt"))
    again = _session(["status", "gen the "], d)                        # a brand new process
    assert "step=30" in again, "training progress must carry over a restart"
    assert "[Model not trained yet]" not in again
    assert json.load(open(os.path.join(d, "carbide_settings.json")))["model.d_model"] == 16


def test_bad_input_prints_an_error_and_the_session_survives():
    d = tempfile.mkdtemp()
    out = _session(["train abc", "set nosuch 1", "set gen.top_p 7", "graph ingest onlyone", "load nosuchsuffix", "nonsense", "status"], d)
    assert out.count("error:") >= 1 and "no setting 'nosuch'" in out and "must be between" in out
    assert "unknown command" in out and "kind=" in out, "the shell must still answer after every bad command"


def test_settings_change_real_behaviour():
    d = tempfile.mkdtemp()
    out = _session(_small() + ["train 20", "set gen.seed 5", "set gen.temperature 0", "gen the", "gen the",
                               "preset wild", "set gen.top_p", "set train.learning_rate 0.01", "set gen.length 10", "gen the"], d)
    lines = [ln for ln in out.splitlines() if ln.startswith("the")]
    assert len(lines) >= 3 and lines[0] == lines[1], "greedy + a fixed seed must repeat"
    assert "preset wild" in out and "gen.top_p = 0.98" in out
    assert "train.learning_rate = 0.01" in out and len(lines[-1]) <= len("the") + 10 + 1


def test_set_json_is_what_the_tui_reads():
    out = _session(["set json"], tempfile.mkdtemp())
    data = json.loads([ln for ln in out.splitlines() if ln.startswith("{")][0])
    names = {s["name"] for s in data["settings"]}
    assert {"model.kind", "train.learning_rate", "gen.top_p", "graph.accept_confidence"} <= names
    assert isinstance(data["notes"], list)


def test_graph_commands_dump_data_and_report_it():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "law.tsv"), "w") as f:
        f.write("subject\trelation\tobject\nplaintiff\tsues\tdefendant\n")
    out = _session(["graph stats", "graph ingest law law.tsv triples", "graph stats", "graph report plaintiff"], d)
    assert "no graph database" in out and '"added": 1' in out
    assert "IN_DOMAIN" in out and "[law/import/accepted]" in out and '"modules": ["core", "law"]' in out
