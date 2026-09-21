"""Graph memory for Carbide: a typed, evidence-weighted knowledge graph that grows
and retunes itself automatically. See store.py for the rules it enforces,
compile.py for the table Carbide reads, ingest.py for dumping domain data in."""
from .compile import CompiledTable, N_GRAPH_COLS, compile_for_vocab, grow_embedding, sync_embedding  # noqa: F401
from .store import GraphStore  # noqa: F401
