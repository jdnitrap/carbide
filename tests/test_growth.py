"""Automatic depth growth: exactly function-preserving, gated on held-out loss, and rolled back exactly."""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from carbide_modules import dataset, growth, layers, training  # noqa: E402
from carbide_modules.config import config  # noqa: E402
from carbide_modules.graphmem import N_GRAPH_COLS  # noqa: E402
from carbide_modules.state import train_state  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "carbide_training_dataset.txt")


def _setup(kind="beside"):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "c.txt")
    with open(CORPUS, "rb") as f, open(path, "wb") as out:
        out.write(f.read(80_000))
    layers.WORD_VOCAB_PATH = os.path.join(tmp, "word_vocab.json")
    layers._reset_vocab_cache()
    config.d_model, config.n_layers, config.d_state = 16, 2, 8
    config.seq_len, config.batch_size, config.model_kind = 32, 4, kind
    config.checkpoint_dir = os.path.join(tmp, "ck")
    config.mdbe_snapshot_dir = os.path.join(tmp, "snap")
    config.mdbe_snapshot_interval = 10**9
    train_state.reset()
    training.model = training.opt = None
    dataset.load_dataset(path)
    return tmp


def _x():
    return torch.tensor([[ord(c) for c in "the cat sat on the mat. a dog ran."]])


def test_growing_depth_leaves_the_output_exactly_unchanged_for_every_model_kind():
    for kind in ("beside", "layered"):
        _setup(kind)
        model = training._build_model(kind).eval()
        with torch.no_grad():
            before = model(_x())
            growth.grow_depth(model, 2)
            after = model(_x())
        assert len(model.blocks) == 4 and torch.allclose(before, after, atol=1e-6), kind


def test_a_grown_graph_model_is_unchanged_too():
    _setup("beside")
    layers.WORD_VOCAB_PATH = os.path.join(tempfile.mkdtemp(), "v.json")
    layers._write_vocab_file(["the", "cat", "sat"])
    layers._reset_vocab_cache()
    m = layers.Carbide(d_model=16, n_layers=1, d_state=8, graph_cols=N_GRAPH_COLS, default_mode="l1_l2_graph").eval()
    table = torch.zeros(layers.WORD_VOCAB_SIZE, N_GRAPH_COLS)
    table[1:4] = torch.rand(3, N_GRAPH_COLS)
    m.set_graph_table(table)
    with torch.no_grad():
        before = m(_x())
        growth.grow_depth(m)
        assert torch.allclose(before, m(_x()), atol=1e-6)


def test_the_new_block_actually_trains():
    _setup()
    training.init_model()
    for _ in range(5):
        training.train_step()
    new = growth.grow_depth(training.model)
    growth._extend_optimizer()
    w0 = new[0].ssm.W_C.weight.clone()
    for _ in range(5):
        training.train_step()
    assert not torch.equal(new[0].ssm.W_C.weight, w0), "the identity-initialised block must receive gradient"


def test_plateau_detection():
    assert not growth.plateaued([2.0] * 50, window=100)
    assert growth.plateaued([2.0] * 200, window=100, min_gain=0.01)
    assert not growth.plateaued([3.0] * 100 + [2.0] * 100, window=100, min_gain=0.01)


def test_a_growth_that_does_not_help_is_rolled_back_exactly_and_logged():
    _setup()
    training.init_model()
    growth.dataset.set_holdout(0.05)
    for _ in range(10):
        training.train_step()
    train_state.step, train_state.loss_history = 10, [(i, 2.0) for i in range(1, 11)]
    weights = [p.detach().clone() for p in training.model.parameters()]
    calls = iter([2.0, 2.5])                                   # held-out gets WORSE after growth
    real = growth.heldout_loss
    growth.heldout_loss = lambda model, **k: next(calls)
    log = os.path.join(tempfile.mkdtemp(), "g.jsonl")
    try:
        rec = growth.try_growth(probation=5, log_path=log)
    finally:
        growth.heldout_loss = real
    assert rec["kept"] is False and len(training.model.blocks) == 2 and config.n_layers == 2
    assert train_state.step == 10 and len(train_state.loss_history) == 10
    assert all(torch.equal(a, b) for a, b in zip(weights, training.model.parameters())), "weights must be restored exactly"
    assert json.loads(open(log).read().splitlines()[-1])["kept"] is False


def test_a_growth_that_helps_is_kept_and_survives_a_checkpoint_round_trip():
    _setup()
    training.init_model()
    dataset.set_holdout(0.05)
    for _ in range(5):
        training.train_step()
    calls = iter([2.0, 1.5])                                   # held-out improves
    real = growth.heldout_loss
    growth.heldout_loss = lambda model, **k: next(calls)
    try:
        rec = growth.try_growth(probation=5, log_path=os.path.join(tempfile.mkdtemp(), "g.jsonl"))
    finally:
        growth.heldout_loss = real
    assert rec["kept"] and len(training.model.blocks) == 3 and config.n_layers == 3
    training.save_checkpoint()
    n_params = sum(p.numel() for p in training.model.parameters())
    training.model = training.opt = None
    config.n_layers = 2                                         # the checkpoint's own depth must win
    assert training.load_checkpoint()
    assert len(training.model.blocks) == 3 and sum(p.numel() for p in training.model.parameters()) == n_params


def test_training_never_touches_the_reserved_held_out_tail():
    _setup()
    dataset.data = torch.arange(20_000)                       # every element equals its own index
    start = dataset.set_holdout(0.05)
    assert start == 19_000
    for _ in range(200):
        x, y = dataset.get_batch(batch=8, seq_len=64)
        assert int(y.max()) < start, "a training window reached into the held-out tail"
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        x, y = dataset.get_holdout_batch(g, batch=8, seq_len=64)
        assert int(x.min()) >= start, "a held-out window strayed into the training part"


def test_the_optimizer_keeps_its_momentum_across_a_growth():
    _setup()
    training.init_model()
    for _ in range(5):
        training.train_step()
    old_params = list(training.model.parameters())
    before = {id(p): training.opt.state[p]["exp_avg"].clone() for p in old_params}
    growth.grow_depth(training.model)
    growth._extend_optimizer()
    assert len(training.opt.param_groups) == 1
    assert all(torch.equal(training.opt.state[p]["exp_avg"], before[id(p)]) for p in old_params)
