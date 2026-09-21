"""Interactive terminal UI for Carbide -- same layout as esgm-gru's TUI.

  python -m carbide_modules.tui [--data corpus.txt]      (needs: pip install textual)

Left: live model / training / graph status, with the editable settings panel beside it.
Right: scrolling log and a command line. carbide_modules.shell runs unmodified in a worker
thread (its stdin is a queue, its print is captured), so every command is a shell command.
After each command the panels refresh by silently running the read-only commands
`status` and `set json`.

Keys: Up/Down history | Right accepts the inline completion | F1 help | F2 settings panel
(Enter edits a value, or cycles a choice) | F5 train 100 | F8 stop training | F9 save |
Ctrl+L clear log | Ctrl+Q quit.
"""
import json
import queue
import re
import threading
import traceback
import types
from collections import deque

from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.suggester import Suggester
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

POLLS = ("status", "set json")   # "set json" must stay last: its arrival triggers the panel redraw
COMMANDS = ("help", "status", "train", "reset", "generate", "preset", "set", "save", "load", "checkpoints",
            "data", "graph", "teacher", "quit")
GRAPH_SUBS = ("stats", "build-core", "ingest", "import-dimensions", "teach", "facts", "fact-corpus", "refresh", "report")
PRESETS = ("calm", "balanced", "wild")
FORMATS = ("triples", "glossary", "text")
WHERE_MARK = {"file": "f", "graph": "g"}
GROUP_STYLE = {"model": "cyan", "train": "yellow", "gen": "magenta", "graph": "green"}

# the shell reports these itself, but catching the obvious cases here saves a round trip
MIN_ARGS = {"preset": 1, "data": 1}
INT_ARGS = {"train": (0,)}


def fmt_value(v):
    if v is None:
        return "off"
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class ShellBridge:
    """Runs carbide_modules.shell.main() in a daemon thread. It is also the shell's stdin: the shell
    asks for the next line only when the previous command finished, so each __next__ call is the
    'command done' signal that flushes the captured output."""

    def __init__(self, app, data_path=None):
        self._app = app
        self._data = data_path
        self._inbox = queue.Queue()
        self._pending = deque()
        self._buf = []
        self._current = None  # (line, silent)
        self._shell = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, line):
        self._inbox.put(line)

    def stop_training(self):
        if self._shell is not None:
            self._shell.STOP.set()

    def close(self):
        self.stop_training()
        self._inbox.put(None)
        self._thread.join(timeout=3)

    def _emit(self, fn, *args):
        try:
            self._app.call_from_thread(fn, *args)
        except Exception:
            pass  # app already shutting down

    # -- the shell's `print` and `sys.stdin`, injected as module attributes --
    def _print(self, *args, sep=" ", end="\n", **_):
        text = sep.join(str(a) for a in args)
        if self._current is not None and not self._current[1]:
            self._emit(self._app.on_stream, text)   # a person's command: show output as it happens
        else:
            self._buf.append(text)                  # a silent poll: collect it and parse at the end

    def __iter__(self):
        return self

    def __next__(self):
        self._finish_current()
        if self._pending:
            line, silent = self._pending.popleft()
        else:
            self._emit(self._app.set_busy, None)
            line = self._inbox.get()
            if line is None:
                raise StopIteration
            silent = False
            self._emit(self._app.set_busy, line)
        self._current = (line, silent)
        return line + "\n"

    def _finish_current(self):
        if self._current is None:
            return
        line, silent = self._current
        self._current = None
        text = "\n".join(self._buf)
        self._buf.clear()
        if silent:
            self._emit(self._app.on_poll, line, text)
        else:
            self._emit(self._app.on_output, line, text)
            self._pending.extend((c, True) for c in POLLS)

    def _run(self):
        try:
            from carbide_modules import dataset, generation, layers, shell, training
        except Exception:
            self._emit(self._app.on_fatal, traceback.format_exc())
            return
        self._shell = shell
        for mod in (shell, training, dataset, layers, generation):
            mod.print = self._print   # capture their output instead of writing to the terminal
        shell.sys = types.SimpleNamespace(stdin=self)
        self._pending.extend((c, True) for c in POLLS)
        try:
            shell.main(self._data)
        except Exception:
            self._buf.append("shell stopped:\n" + traceback.format_exc(limit=-2).rstrip())
            self._finish_current()
            self._emit(self._app.on_fatal, "the shell stopped unexpectedly (see the log)")
            return
        self._emit(self._app.on_shell_exit)


class CommandSuggester(Suggester):
    """Inline completion: command names, then settings / presets / graph subcommands."""

    def __init__(self, app):
        super().__init__(use_cache=False, case_sensitive=True)
        self._app = app

    async def get_suggestion(self, value):
        parts = value.split(" ")
        last = parts[-1]
        if not last:
            return None
        if len(parts) == 1:
            candidates = COMMANDS
        elif parts[0] == "set" and len(parts) == 2:
            candidates = [s["name"] for s in self._app.settings]
        elif parts[0] == "set" and len(parts) == 3:
            row = next((s for s in self._app.settings if s["name"] == parts[1]), None)
            candidates = (row["choices"] or (("off",) if row["nullable"] else ())) if row else ()
        elif parts[0] == "preset" and len(parts) == 2:
            candidates = PRESETS
        elif parts[0] == "graph" and len(parts) == 2:
            candidates = GRAPH_SUBS
        elif parts[0] == "graph" and parts[1] == "ingest" and len(parts) == 5:
            candidates = FORMATS
        else:
            return None
        for c in candidates:
            if c.startswith(last) and c != last:
                return " ".join(parts[:-1] + [c])
        return None


class HistoryInput(Input):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hist = []
        self._pos = 0

    def remember(self, line):
        if not self._hist or self._hist[-1] != line:
            self._hist.append(line)
        self._pos = len(self._hist)

    def on_key(self, event: events.Key):
        if event.key == "up" and self._pos > 0:
            self._pos -= 1
        elif event.key == "down" and self._pos < len(self._hist):
            self._pos += 1
        else:
            return
        self.value = self._hist[self._pos] if self._pos < len(self._hist) else ""
        self.cursor_position = len(self.value)
        event.stop()
        event.prevent_default()


class CarbideApp(App):
    TITLE = "CARBIDE"
    CSS = """
    #left { width: 46; }
    #status { border: round $primary; padding: 0 1; height: auto; }
    #settings { height: 1fr; border: round $primary; }
    #main { width: 1fr; }
    #log { border: round $primary; }
    #cmd { dock: bottom; }
    """
    BINDINGS = [
        Binding("f1", "cmd('help')", "Help"),
        Binding("f2", "focus_settings", "Settings"),
        Binding("f5", "cmd('train 100')", "Train 100"),
        Binding("f8", "stop_training", "Stop"),
        Binding("f9", "cmd('save')", "Save"),
        Binding("escape", "focus_input", "Input", show=False),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, data_path=None):
        super().__init__()
        self.stats = {}
        self.settings = []
        self.bridge = ShellBridge(self, data_path)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                status = Static("loading...", id="status")
                status.border_title = "Carbide"
                yield status
                table = DataTable(id="settings", cursor_type="row", zebra_stripes=True)
                table.border_title = "Settings"
                table.border_subtitle = "Enter=edit  f=auto-saved g=graph db"
                yield table
            with Vertical(id="main"):
                log = RichLog(id="log", wrap=True, markup=False, highlight=False)
                log.border_title = "Log"
                yield log
        yield HistoryInput(id="cmd", placeholder="command (F1 for help)", suggester=CommandSuggester(self))
        yield Footer()

    def on_mount(self):
        self.query_one("#settings", DataTable).add_columns("setting", "value", "")
        self.query_one("#cmd").focus()
        self.sub_title = "loading..."
        self.bridge.start()
        self.set_interval(1.0, self._live_status)

    def on_unmount(self):
        self.bridge.close()

    # -- input ---------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted):
        line = event.value.strip()
        event.input.value = ""
        if not line:
            return
        event.input.remember(line)
        self.action_cmd(line)

    def action_cmd(self, line):
        self._log(Text(f"> {line}", style="bold cyan"))
        problem = self._validate(line)
        if problem:
            self._log(Text(problem, style="red"))
        else:
            self.bridge.submit(line)

    def action_stop_training(self):
        self.bridge.stop_training()
        self._log(Text("stop requested -- training ends after the current step", style="yellow"))

    def action_clear_log(self):
        self.query_one("#log", RichLog).clear()

    def action_focus_settings(self):
        self.query_one("#settings", DataTable).focus()

    def action_focus_input(self):
        self.query_one("#cmd", Input).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Enter on a setting: cycle a choice right away, otherwise load `set <name> <value>` into the
        input for editing, so every change goes through the shell's own validation."""
        name = event.row_key.value
        row = next((s for s in self.settings if s["name"] == name), None)
        if row is None:
            return
        if row["kind"] == "bool":
            self.action_cmd(f"set {name} {'off' if row['value'] else 'on'}")
            return
        if row["choices"]:
            nxt = row["choices"][(row["choices"].index(row["value"]) + 1) % len(row["choices"])]
            self.action_cmd(f"set {name} {nxt}")
            return
        inp = self.query_one("#cmd", Input)
        inp.value = f"set {name} {fmt_value(row['value'])}"
        inp.cursor_position = len(inp.value)
        inp.focus()

    def _validate(self, line):
        cmd, *args = line.split()
        if len(args) < MIN_ARGS.get(cmd, 0):
            return f"{cmd} needs {MIN_ARGS[cmd]} argument(s) -- see help"
        for i in INT_ARGS.get(cmd, ()):
            if i < len(args) and not args[i].lstrip("-").isdigit():
                return f"{cmd}: argument {i + 1} must be an integer, got {args[i]!r}"
        return None

    # -- results from the worker thread (always via call_from_thread) --------
    def set_busy(self, line):
        self.sub_title = f"running: {line}" if line else "ready"

    def on_output(self, line, text):
        if text:
            bad = text.startswith(("error:", "shell stopped"))
            self._log(Text(text, style="red" if bad else ""))

    def on_stream(self, text):
        self._log(Text(text, style="red" if text.startswith("error:") else ""))

    def _live_status(self):
        """While a train command runs, refresh the status panel once a second (reads only)."""
        if not self.sub_title.startswith("running: train") or self.bridge._shell is None:
            return
        try:
            self.stats = dict(re.findall(r"(\w+)=(\S+)", self.bridge._shell.status_line()))
        except Exception:
            return
        self._render_status()

    def on_poll(self, line, text):
        try:
            if line == "status":
                self.stats = dict(re.findall(r"(\w+)=(\S+)", text))
            elif line == "set json":
                data = json.loads(text)
                self.settings = data["settings"]
                for note in data["notes"]:
                    self._log(Text(note, style="yellow"))
        except (ValueError, KeyError):
            return
        if line == "set json":
            self._render_status()
            self._render_settings()

    def on_fatal(self, message):
        self._log(Text(message, style="bold red"))
        self.sub_title = "shell not running"
        self.query_one("#cmd").disabled = True

    def on_shell_exit(self):
        self.exit()

    # -- rendering -----------------------------------------------------------
    def _log(self, renderable):
        self.query_one("#log", RichLog).write(renderable)

    def _render_status(self):
        self.query_one("#status", Static).update(self._side_table())

    def _render_settings(self):
        table = self.query_one("#settings", DataTable)
        keep = table.cursor_row
        table.clear()
        for row in self.settings:
            group = row["name"].split(".", 1)[0]
            table.add_row(Text(row["name"], style=GROUP_STYLE.get(group, "")), fmt_value(row["value"]),
                          Text(WHERE_MARK.get(row["where"], "?"), style="dim"), key=row["name"])
        if self.settings:
            table.move_cursor(row=min(keep, len(self.settings) - 1))

    def _side_table(self):
        s = self.stats
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim")
        grid.add_column()
        if not s:
            grid.add_row("status", "unavailable")
            return grid
        try:
            grid.add_row("model", f"{s['kind']}  {int(s['params']):,} params")
            grid.add_row("shape", f"d={s['d_model']} layers={s['n_layers']} state={s['d_state']}")
            grid.add_row("step", f"{s['step']}   loss {s['loss']}")
            grid.add_row("batch", f"{s['batch']} x {s['seq_len']}  lr {s['lr']}")
            grid.add_row("data", f"{s['data']} ({int(s['bytes']):,} B)")
            grid.add_row("vocab", f"{s['vocab']} words")
            if s["graph"] == "on":
                grid.add_row("graph", f"{int(s['nodes']):,} nodes  {int(s['edges']):,} edges")
                grid.add_row("", f"{s['modules']} modules  {s['dims']} dims  cap {int(s['capacity']):,}")
            else:
                grid.add_row("graph", "not built  (graph build-core)")
        except KeyError:
            grid.add_row("status", "unavailable")
        return grid


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Carbide TUI")
    ap.add_argument("--data", default=None, help="training text file")
    CarbideApp(ap.parse_args().data).run()


if __name__ == "__main__":
    main()
