"""Auto-add discovered dims as proposed; fix_dimensions can lock them."""
import os
import tempfile

from carbide_modules.discover_dimension import _persist, fix_dimensions, load_discovered


def test_auto_add_is_proposed_beside_metadata():
    path = os.path.join(tempfile.mkdtemp(), "d.json")
    entry = _persist("DISCOVERED_NOUN_LIKE", "NOUN_LIKE",
                     ["cat", "dog", "house"] * 4, path)
    assert entry["status"] == "proposed"
    assert entry["confidence"] == 0.20


def test_fix_marks_large_cluster_fixed():
    td = tempfile.mkdtemp()
    path = os.path.join(td, "d.json")
    words = [f"w{i}" for i in range(12)]
    _persist("DISCOVERED_NOUN_LIKE", "NOUN_LIKE", words, path)
    corpus = os.path.join(td, "c.txt")
    open(corpus, "w").write(" ".join(words * 30))
    report = fix_dimensions(corpus, model=None, path=path, min_keep_freq=1)
    assert report[0]["status"] == "fixed"
    assert report[0]["confidence"] >= 0.35
    saved = load_discovered(path)
    assert saved["DISCOVERED_NOUN_LIKE"]["status"] == "fixed"
