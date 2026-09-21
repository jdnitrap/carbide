"""python -m carbide_modules.graphmem <command>

  build-core         build the dictionary module from a corpus + WordNet
  ingest             dump triples / a glossary / raw text into a domain module
  compile            write the lookup table Carbide reads
  import-dimensions  move discovered_dimensions.json into the graph
  report             what the graph holds (add --word W for one word)
"""
import argparse
import json
import sys

from . import ingest as ing
from .compile import compile_for_vocab
from .store import GraphStore
from .wordnet_loader import build_core


def _vocab_file(path):
    try:
        with open(path) as f:
            return list(json.load(f).get("words") or [])
    except FileNotFoundError:
        return []


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m carbide_modules.graphmem")
    p.add_argument("--db", default="graph_memory.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-core")
    b.add_argument("--corpus", default="carbide_training_dataset.txt")
    b.add_argument("--vocab-size", type=int, default=5000)
    b.add_argument("--min-count", type=int, default=2)
    b.add_argument("--vocab-file", default="word_vocab.json", help="also add Carbide's Layer-2 vocabulary")
    i = sub.add_parser("ingest")
    i.add_argument("--module", required=True)
    i.add_argument("--file", required=True)
    i.add_argument("--format", choices=ing.FORMATS, required=True)
    c = sub.add_parser("compile")
    c.add_argument("--out", default="graph_table.pt")
    c.add_argument("--vocab-file", default="word_vocab.json")
    c.add_argument("--rows", type=int, default=2048)
    c.add_argument("--modules", default=None, help="comma-separated modules to include (default: all enabled)")
    d = sub.add_parser("import-dimensions")
    d.add_argument("--file", default="discovered_dimensions.json")
    r = sub.add_parser("report")
    r.add_argument("--word")
    a = p.parse_args(argv)

    with GraphStore(a.db) as st:
        if a.cmd == "build-core":
            out = build_core(st, a.corpus, vocab_size=a.vocab_size, min_count=a.min_count,
                             extra_words=_vocab_file(a.vocab_file))
        elif a.cmd == "ingest":
            out = ing.dump(st, a.module, a.file, a.format)
        elif a.cmd == "compile":
            words = _vocab_file(a.vocab_file)
            mods = a.modules.split(",") if a.modules else None
            t = compile_for_vocab(st, words, a.rows, modules=mods)
            t.save(a.out)
            out = {"words": len(words), "rows": a.rows, "columns": t.table.shape[1], "saved": a.out,
                   "known_words": int((t.table[:, :4].sum(-1) > 0).sum())}
        elif a.cmd == "import-dimensions":
            out = ing.import_discovered_json(st, a.file)
        else:
            out = st.stats()
            if a.word:
                out = {"word": a.word, "edges": st.edges_of(a.word), "note": (st.node(st.node_id(a.word)) or {}).get("note")}
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
