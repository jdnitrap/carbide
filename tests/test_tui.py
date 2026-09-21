"""Drive the real TUI headless: panels fill, commands run, settings edit, training stops on demand.
Skips (passes) when textual is not installed:  pip install textual"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(os.path.dirname(__file__), "..", "carbide_training_dataset.txt")


def _run(coro):
    return asyncio.run(coro)


async def _session():
    from textual.widgets import DataTable, Input, RichLog

    from carbide_modules import dataset, generation, layers, shell, training
    from carbide_modules.tui import CarbideApp

    tmp, old = tempfile.mkdtemp(), os.getcwd()
    with open(CORPUS, "rb") as f, open(os.path.join(tmp, "corpus.txt"), "wb") as out:
        out.write(f.read(60_000))
    os.chdir(tmp)
    layers.WORD_VOCAB_PATH = os.path.join(tmp, "word_vocab.json")
    layers._reset_vocab_cache()
    app = CarbideApp("corpus.txt")
    log = lambda: "\n".join(s.text for s in app.query_one("#log", RichLog).lines)  # noqa: E731
    result = {}
    try:
        async with app.run_test(size=(140, 44)) as pilot:
            async def ready(limit=120):
                for _ in range(limit * 10):
                    await pilot.pause(0.1)
                    if app.sub_title == "ready":
                        return
                raise AssertionError("never became ready")

            async def send(text):
                app.query_one("#cmd", Input).value = text
                await pilot.press("enter")
                await pilot.pause(0.3)
                await ready()

            await pilot.pause(0.5)
            await ready()
            table, inp = app.query_one("#settings", DataTable), app.query_one("#cmd", Input)
            result["rows"], result["polled"] = table.row_count, len(app.settings)
            result["kind_at_start"] = app.stats.get("kind")
            for c in ("set model.d_model 16", "set model.n_layers 1", "set model.d_state 8",
                      "set train.seq_len 32", "set train.batch_size 4"):
                await send(c)
            await send("train 20")
            result["step"] = app.stats.get("step")
            await send("gen the")
            await pilot.press("f2")
            result["f2_focus"] = app.focused is table
            names = [s["name"] for s in app.settings]
            table.move_cursor(row=names.index("model.kind"))
            await pilot.press("enter")
            await pilot.pause(0.3)
            await ready()
            result["kind_cycled"] = next(s["value"] for s in app.settings if s["name"] == "model.kind")
            table.move_cursor(row=names.index("gen.top_p"))
            await pilot.press("enter")
            await pilot.pause(0.2)
            result["prefill"] = inp.value
            result["input_focus"] = app.focused is inp
            inp.value = "set gen.top_p 0.5"
            await pilot.press("enter")
            await pilot.pause(0.3)
            await ready()
            result["top_p"] = next(s["value"] for s in app.settings if s["name"] == "gen.top_p")
            await send("set model.kind beside")
            await send("reset")
            await send("set gen.top_p ")             # (a partial command must be handled, not crash)
            # stop a long run from the keyboard
            app.query_one("#cmd", Input).value = "train 20000"
            await pilot.press("enter")
            for _ in range(300):                                          # wait for visible progress (a busy machine is slow)
                await pilot.pause(0.1)
                if int(app.stats.get("step", 0)) > 0:
                    break
            result["live_step"] = int(app.stats.get("step", 0))          # before the command has finished
            result["live_log"] = log()
            app.action_stop_training()
            await ready()
            result["stopped_step"] = int(app.stats.get("step", 0))
            result["log"] = log()
            inp.value = "set gen.top"
            await pilot.pause(0.4)
            result["suggest"] = inp._suggestion
    finally:
        os.chdir(old)
        for mod in (shell, training, dataset, layers, generation):
            if "print" in vars(mod):
                delattr(mod, "print")
        if hasattr(shell, "sys"):
            delattr(shell, "sys")
    return result


def test_the_tui_drives_carbide_end_to_end():
    try:
        import textual  # noqa: F401
    except ImportError:
        print("SKIP: textual not installed")
        return
    r = _run(_session())
    n_settings = 25
    assert r["rows"] == r["polled"] == n_settings, r
    assert r["kind_at_start"] == "beside"
    assert r["step"] == "20", r["step"]
    assert r["f2_focus"] and r["input_focus"]
    assert r["kind_cycled"] == "layered", "Enter on a choice setting should cycle it"
    assert r["prefill"] == "set gen.top_p 1" and r["top_p"] == 0.5
    assert r["stopped_step"] < 20000 and "Stopped early" in r["log"], (r["stopped_step"])
    assert r["live_step"] > 0, "the status panel should move while training is still running"
    assert "step " in r["live_log"].split("> train 20000")[-1], "output should stream before the command ends"
    assert "> train 20" in r["log"] and "Checkpoint saved" in r["log"] and "gen.top_p = 0.5" in r["log"]
    assert r["suggest"] == "set gen.top_k" or r["suggest"] == "set gen.top_p", r["suggest"]
