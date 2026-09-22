"""Line-command shell for Carbide -- the same idea as esgm-gru's shell.py, and what the TUI wraps.

Reads one command per line from stdin and dispatches to the existing training / generation /
checkpoint / graph code. It adds no logic of its own; every command is wrapped so a bad input
prints an error and the session (and any unsaved model state) survives.

  python -m carbide_modules.shell [--data corpus.txt]
"""
import argparse
import json
import os
import sys
import threading

from . import dataset, generation, layers, settings, training
from .config import config
from .state import train_state

STOP = threading.Event()   # set from another thread (the TUI's Stop key) to end a `train` run cleanly
NOTES = []                 # startup notices, shown once by `set`

HELP_TEXT = """help
status                       one-line summary (model, training, data, graph)
train [n=100]                train n steps (checkpoint is saved when the run ends)
grow words [rows]            add word-table rows (also automatic when headroom runs low)
reset                        forget the current model and training progress
generate [+facts] <prompt...> generate text with the gen.* settings (+facts puts the graph's facts in front; \\n = newline; alias: gen)
preset <calm|balanced|wild>  set all the sampling controls at once
set [name [value]]           list / show / change any knob (`set json` = the raw table)
teacher <model> <genre> <n> <out> [topics...] --license-ok   a local Ollama model writes training text; the graph filters it
save [suffix]                save a checkpoint          load [suffix] [migrate]   load one
checkpoints                  list saved checkpoints
data <path>                  load a training text file
graph stats                  what the graph memory holds
graph build-core [corpus]    build the dictionary module from a corpus + WordNet (needs nltk)
graph ingest <module> <file> <triples|glossary|text>   dump domain data into a module
graph import-dimensions [file]   move discovered_dimensions.json into the graph
graph facts <words...>        the graph's facts about words, as sentences
graph fact-corpus <in> <out> write a training corpus with facts in front of each sentence
graph teach [corpus]         Carbide teaches the graph (probe on its hidden states; gated, automatic)
graph refresh                re-read the graph into the live graph model
graph report <word>          everything the graph knows about a word
quit"""


def _store(create=False):
    from .graphmem import GraphStore
    if not create and not os.path.exists(config.graph_db):
        return None
    return GraphStore(config.graph_db)


def status_line():
    m = training.model
    p = {
        "kind": training._kind_of(m) if m is not None else config.model_kind,
        "params": sum(x.numel() for x in m.parameters()) if m is not None else 0,
        "step": train_state.step,
        "loss": f"{train_state.current_loss:.4f}" if train_state.current_loss else "n/a",
        "d_model": config.d_model, "n_layers": config.n_layers, "d_state": config.d_state,
        "seq_len": config.seq_len, "batch": config.batch_size, "lr": f"{config.learning_rate:g}",
        "device": f"{config.device_pref}({config.resolved_device()})",
        "data": (os.path.basename(config.data_file) if config.data_file else "default").replace(" ", "_"),
        "bytes": len(dataset.data) if dataset.data is not None else 0,
        "vocab": len(layers.word_vocab()[0]), "word_rows": layers.WORD_ROWS,
        "graph": "off", "nodes": 0, "edges": 0, "modules": 0, "capacity": 0, "dims": 0,
    }
    st = _store()
    if st is not None:
        try:
            s = st.stats()
            p.update(graph="on", nodes=s["nodes"], edges=s["edges"], modules=len(s["modules"]),
                     capacity=s["capacity"], dims=s["dimensions"])
        finally:
            st.close()
    return " ".join(f"{k}={v}" for k, v in p.items())


def _gen_kwargs():
    g = lambda name, default: getattr(config, name, default)  # noqa: E731
    d = generation.DEFAULTS
    return dict(temperature=g("gen_temperature", d[0]), top_k=g("gen_top_k", d[1]), top_p=g("gen_top_p", d[2]),
                repetition_penalty=g("gen_repetition_penalty", d[3]), no_repeat_ngram=g("gen_no_repeat_ngram", d[4]),
                seed=g("gen_seed", None))


def _graph(args):
    sub = args[0] if args else "stats"
    if sub == "stats":
        st = _store()
        if st is None:
            print(f"no graph database at {config.graph_db!r} -- run: graph build-core")
            return
        try:
            s = st.stats()
        finally:
            st.close()
        print(json.dumps({k: s[k] for k in ("nodes", "edges", "capacity", "dimensions", "modules", "runs", "policy")}))
        print("edges by relation:", json.dumps(s["edges_by_relation"]))
    elif sub == "build-core":
        from .graphmem.wordnet_loader import build_core
        corpus = args[1] if len(args) > 1 else (config.data_file or "carbide_training_dataset.txt")
        if not os.path.exists(corpus):
            print(f"no corpus at {corpus!r}")
            return
        with _store(create=True) as st:
            print("built:", json.dumps(build_core(st, corpus, extra_words=layers.word_vocab()[0])))
    elif sub == "ingest":
        from .graphmem import ingest
        if len(args) != 4 or args[3] not in ingest.FORMATS:
            print(f"usage: graph ingest <module> <file> <{'|'.join(ingest.FORMATS)}>")
            return
        if not os.path.exists(args[2]):
            print(f"no such file {args[2]!r}")
            return
        with _store(create=True) as st:
            print("ingested:", json.dumps(ingest.dump(st, args[1], args[2], args[3])))
    elif sub == "import-dimensions":
        from .graphmem import ingest
        path = args[1] if len(args) > 1 else "discovered_dimensions.json"
        if not os.path.exists(path):
            print(f"no such file {path!r}")
            return
        with _store(create=True) as st:
            print("imported:", json.dumps(ingest.import_discovered_json(st, path)))
    elif sub == "teach":
        from .graphmem import teach
        if training.model is None:
            print("no model yet -- `train` or `load` first")
            return
        st = _store()
        if st is None:
            print("no graph database yet -- run: graph build-core")
            return
        try:
            if len(args) > 1:
                with open(args[1], "rb") as f:
                    raw = f.read(400_000)
            else:
                raw = bytes(dataset.data[:400_000].tolist())
            words = [r[0] for r in st.db.execute("SELECT text FROM nodes WHERE kind = 'WORD'")]
            print(f"reading Carbide's hidden states for {len(words)} dictionary words...")
            vectors = teach.collect_word_vectors(training.model, raw, words)
            print("teach:", json.dumps(teach.probe_and_propose(st, vectors)))
        finally:
            st.close()
    elif sub == "facts":
        from .graphmem import retrieval
        if len(args) < 2:
            print("usage: graph facts <words...>")
            return
        st = _store()
        if st is None:
            print("no graph database yet")
            return
        try:
            facts = retrieval.retrieve(st, " ".join(args[1:]), max_words=4, per_word=3)
        finally:
            st.close()
        print("\n".join(facts) or "(no facts)")
    elif sub == "fact-corpus":
        from .graphmem import retrieval
        if len(args) != 3 or not os.path.exists(args[1]):
            print("usage: graph fact-corpus <input.txt> <output.txt>")
            return
        with open(args[1], encoding="utf-8", errors="replace") as f:
            text = f.read()
        st = _store()
        if st is None:
            print("no graph database yet")
            return
        try:
            total, withf = retrieval.build_fact_corpus(st, text, args[2])
        finally:
            st.close()
        print(f"wrote {args[2]}: {total} sentences, {withf} with facts (train on it: data {args[2]})")
    elif sub == "refresh":
        training.refresh_graph_table()
        print("graph table refreshed into the live model")
    elif sub == "report":
        if len(args) < 2:
            print("usage: graph report <word>")
            return
        st = _store()
        if st is None:
            print("no graph database yet")
            return
        try:
            edges = st.edges_of(args[1].lower())
            node = st.node_id(args[1].lower())
            note = (st.node(node) or {}).get("note") if node else None
        finally:
            st.close()
        print(f"{args[1]}: {note or '(no definition)'}")
        for e in edges:
            print(f"  {e['rel']:<10} {e['dst']:<20} w={e['weight']:.2f} conf={e['confidence']:.2f} "
                  f"[{e['module']}/{e['source']}/{e['status']}]")
    else:
        print("graph: stats | build-core | ingest | import-dimensions | teach | facts | fact-corpus | refresh | report")


def _dispatch(cmd, args):
    if cmd == "help":
        print(HELP_TEXT)
    elif cmd == "status":
        print(status_line())
    elif cmd == "set":
        if not args:
            print(settings.listing())
            while NOTES:
                print(NOTES.pop(0))
        elif args == ["json"]:
            print(json.dumps({"settings": settings.snapshot(), "notes": [NOTES.pop(0) for _ in list(NOTES)]}))
        elif len(args) == 1:
            print(settings.show(args[0]))
        else:
            print(settings.set_setting(args[0], " ".join(args[1:])))
    elif cmd == "train":
        n = int(args[0]) if args else 100
        if n < 1:
            print("train: n must be at least 1")
            return
        if dataset.data is None:
            print("no dataset loaded -- use: data <path>")
            return
        STOP.clear()
        if getattr(config, "train_auto_grow", False):
            from . import growth
            growth.train_with_growth(n, should_stop=STOP.is_set, max_layers=getattr(config, "train_max_layers", 8),
                                     min_gain=getattr(config, "train_growth_min_gain", 0.01))
        else:
            training.train_n_steps(n, should_stop=STOP.is_set)
    elif cmd == "reset":
        train_state.reset()
        training.model = training.opt = None
        print("model and training progress cleared (the next `train` starts fresh)")
    elif cmd in ("generate", "gen", "say"):
        if training.model is None:
            print("[Model not trained yet] -- `train` or `load` first")
            return
        use_facts = bool(args) and args[0] == "+facts"
        prompt = " ".join(args[1:] if use_facts else args).replace("\\n", "\n") or "the "
        prefix = ""
        if use_facts:
            from .graphmem import retrieval
            st = _store()
            if st is None:
                print("(no graph database yet, generating without facts)")
            else:
                try:
                    prefix = retrieval.prefix_for(st, prompt)
                finally:
                    st.close()
                print(prefix.strip() or "(the graph has no facts about these words)")
        text = generation.generate(prefix + prompt, n_bytes=getattr(config, "gen_length", 200), **_gen_kwargs())
        print(text[len(prefix):] if prefix else text)
    elif cmd == "preset":
        if len(args) != 1 or args[0] not in generation.PRESETS:
            print(f"usage: preset <{'|'.join(sorted(generation.PRESETS))}>")
            return
        names = ("temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram")
        for n, v in zip(names, generation.PRESETS[args[0]]):
            settings.set_setting(f"gen.{n}", v)
        print(f"preset {args[0]}: " + ", ".join(f"{n}={v:g}" for n, v in zip(names, generation.PRESETS[args[0]])))
    elif cmd == "save":
        if training.model is None:
            print("nothing to save yet")
            return
        training.save_checkpoint(args[0] if args else "")
    elif cmd == "load":
        ok = training.load_checkpoint(args[0] if args and args[0] != "migrate" else "", allow_migrate="migrate" in args)
        print("loaded" if ok else "not loaded")
    elif cmd == "checkpoints":
        names = sorted(f for f in os.listdir(config.checkpoint_dir) if f.endswith(".pt")) if os.path.isdir(config.checkpoint_dir) else []
        print("\n".join(names) if names else "(none)")
    elif cmd == "data":
        if not args:
            print("usage: data <path>")
            return
        print("loaded" if dataset.load_dataset(" ".join(args)) else "not loaded")
    elif cmd == "grow":
        if not args or args[0] != "words":
            print("usage: grow words [rows]   (depth grows automatically: set train.auto_grow on)")
            return
        from . import layers
        rows = int(args[1]) if len(args) > 1 else layers.WORD_ROWS + training.WORD_ROW_BLOCK
        print(f"word table: {layers.WORD_ROWS} -> {rows} rows" if training.grow_word_table(rows) else "already that big")
    elif cmd == "graph":
        _graph(args)
    elif cmd == "teacher":
        from . import teacher
        ok = "--license-ok" in args
        args = [a for a in args if a != "--license-ok"]
        if len(args) < 4 or args[1] not in teacher.GENRES or not args[2].isdigit():
            print(f"usage: teacher <model> <{'|'.join(teacher.GENRES)}> <count> <out.txt> [topics...] --license-ok")
            return
        STOP.clear()
        st = _store()
        try:
            stats = teacher.run(args[0], args[1], int(args[2]), args[3], license_ok=ok, topics=args[4:],
                                store=st, should_stop=STOP.is_set)
        except teacher.TeacherError as e:
            print(f"teacher: {e}")
            return
        finally:
            if st is not None:
                st.close()
        print(f"teacher: {json.dumps(stats)} -> {args[3]} (manifest: {args[3]}.manifest.jsonl); train on it with: data {args[3]}")
    else:
        print(f"unknown command {cmd!r} -- try: help")


def main(data_path=None):
    NOTES.extend(settings.apply_file_settings())
    if data_path:
        dataset.load_dataset(data_path)
    elif dataset.data is None:
        dataset.load_dataset()
    default_ckpt = f"{config.checkpoint_dir}/carbide_ckpt.pt"
    if os.path.exists(default_ckpt):
        training.load_checkpoint()
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]
        if cmd == "quit":
            break
        try:
            _dispatch(cmd, args)
        except Exception as e:  # one bad command must never end the session
            print(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Carbide command shell")
    ap.add_argument("--data", default=None, help="training text file")
    main(ap.parse_args().data)
