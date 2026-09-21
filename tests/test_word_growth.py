"""Growing the Layer-2 word table: nothing already learned changes, momentum survives, checkpoints keep
their own size, and the automatic rule only fires when headroom runs low."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import dataset, generation, layers, training  # noqa: E402
from carbide_modules.config import config  # noqa: E402
from carbide_modules.graphmem import N_GRAPH_COLS  # noqa: E402
from carbide_modules.state import train_state  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "carbide_training_dataset.txt")
WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran"]


def _setup(kind="layered", rows=64):
    tmp = tempfile.mkdtemp()
    layers.WORD_VOCAB_PATH = os.path.join(tmp, "word_vocab.json")
    layers.set_word_rows(rows)
    layers._write_vocab_file(WORDS)
    config.d_model, config.n_layers, config.d_state = 16, 1, 8
    config.seq_len, config.batch_size, config.model_kind = 32, 4, kind
    config.checkpoint_dir = os.path.join(tmp, "ck")
    config.mdbe_snapshot_dir = os.path.join(tmp, "snap")
    config.mdbe_snapshot_interval = 10**9
    config.graph_db = os.path.join(tmp, "none.db")
    train_state.reset()
    training.model = training.opt = None
    with open(CORPUS, "rb") as f:
        dataset.data = torch.tensor(list(f.read(60_000)), dtype=torch.long)
    return tmp


def _x():
    return torch.tensor([[ord(c) for c in "the cat sat on the mat. a dog ran."]])


def test_growing_the_table_changes_nothing_already_learned():
    _setup()
    m = layers.Carbide(d_model=16, n_layers=1, d_state=8, graph_cols=N_GRAPH_COLS, default_mode="l1_l2_graph").eval()
    table = torch.rand(64, N_GRAPH_COLS)
    m.set_graph_table(table)
    with torch.no_grad():
        before = m(_x())
    old = m.layers.word.weight.detach().clone()
    assert m.grow_word_table(200) and layers.WORD_ROWS == 200
    assert m.layers.word.num_embeddings == 200 and m.layers.graph_table.shape[0] == 200
    assert torch.equal(m.layers.word.weight[:64], old) and torch.equal(m.layers.graph_table[:64], table)
    assert float(m.layers.graph_table[64:].abs().sum()) == 0.0, "new words start with no graph features"
    with torch.no_grad():
        assert torch.equal(before, m(_x())), "the output must be exactly unchanged"
    assert m.grow_word_table(100) is False, "a table never shrinks"


def test_the_vocabulary_can_use_the_new_rows_only_after_growth():
    _setup(rows=10)
    layers._write_vocab_file(WORDS)
    extra = os.path.join(tempfile.mkdtemp(), "e.txt")
    open(extra, "w").write("zzone zztwo zzthree zzfour zzfive zzsix " * 30)
    assert len(layers.build_word_vocab(extra)) == 9, "a 10-row table holds 9 words"
    training.grow_word_table(30)
    assert len(layers.build_word_vocab(extra)) == 13, "after growth the rest of the new words fit"


def test_the_optimizer_keeps_its_momentum_through_a_growth():
    _setup()
    training.init_model()
    for _ in range(5):
        training.train_step()
    old_rows = training.model.layers.word.weight.shape[0]
    moment = training.opt.state[training.model.layers.word.weight]["exp_avg"].clone()
    other = {id(p): training.opt.state[p]["exp_avg"].clone() for p in training.model.parameters()
             if p is not training.model.layers.word.weight and "exp_avg" in training.opt.state[p]}
    assert training.grow_word_table(old_rows + 100)
    new_p = training.model.layers.word.weight
    st = training.opt.state[new_p]["exp_avg"]
    assert st.shape[0] == old_rows + 100 and torch.equal(st[:old_rows], moment) and float(st[old_rows:].abs().sum()) == 0
    assert other and all(torch.equal(training.opt.state[p]["exp_avg"], other[id(p)])
                         for p in training.model.parameters() if id(p) in other)
    for _ in range(3):
        assert torch.isfinite(torch.tensor(training.train_step()))


def test_a_grown_checkpoint_reloads_at_its_own_size_and_an_old_one_at_the_default():
    _setup()
    training.init_model()
    training.grow_word_table(150)
    training.save_checkpoint()
    layers.set_word_rows(64)                                   # a fresh process would start at the default
    training.model = training.opt = None
    assert training.load_checkpoint()
    assert training.model.layers.word.num_embeddings == 150 and layers.WORD_ROWS == 150
    assert isinstance(generation.generate("the ", n_bytes=20, seed=0), str)


def test_a_checkpoint_from_before_word_rows_existed_loads_at_the_default_size():
    _setup(rows=layers.WORD_VOCAB_SIZE)
    training.init_model()
    training.save_checkpoint()
    path = os.path.join(config.checkpoint_dir, "carbide_ckpt.pt")
    ckpt = torch.load(path, weights_only=True)
    del ckpt["word_rows"]                                      # what a checkpoint from before growth existed looks like
    torch.save(ckpt, path)
    layers.set_word_rows(300)
    training.model = training.opt = None
    assert training.load_checkpoint()
    assert training.model.layers.word.num_embeddings == layers.WORD_VOCAB_SIZE == layers.WORD_ROWS


def test_the_automatic_rule_grows_only_when_headroom_runs_low():
    _setup(rows=300)
    training.init_model()
    assert training.ensure_word_capacity() is False, "plenty of room: nothing to do"
    layers._write_vocab_file([f"w{i}" for i in range(280)])    # 19 free rows < the 256 headroom
    assert training.ensure_word_capacity() is True
    assert training.model.layers.word.num_embeddings == 300 + training.WORD_ROW_BLOCK
    assert training.ensure_word_capacity() is False
    training.model = None
    config.model_kind = "beside"
    layers.set_word_rows(300)
    assert training.ensure_word_capacity() is False, "a beside model has no word table to grow"


def test_loading_a_dataset_makes_room_automatically():
    _setup(rows=300)
    training.init_model()
    layers._write_vocab_file([f"w{i}" for i in range(290)])
    path = os.path.join(tempfile.mkdtemp(), "d.txt")
    open(path, "w").write("plain words for a new dataset " * 20)
    assert dataset.load_dataset(path)
    assert training.model.layers.word.num_embeddings == 300 + training.WORD_ROW_BLOCK
