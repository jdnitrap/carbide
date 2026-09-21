"""A local model as Carbide's teacher: it writes text, the graph filters it, Carbide trains on what
survives. (Sequence-level distillation: Carbide learns from the teacher's TEXT, not its scores.)

  ollama serve            # the teacher runs locally; nothing leaves this machine
  teacher <model> <genre> <count> <out.txt> --license-ok [topics...]      (in the shell/TUI)

What this does and does not do:
  * The output is a training text file plus a manifest (one JSON line per attempt: model, digest, prompt,
    seed, kept or the reason it was rejected, sha1). Nothing is hidden: every sample is traceable.
  * Filtering is automatic and cheap -- ASCII only, no markdown, not repetitive, and (with a graph)
    mostly words the dictionary knows. It cannot check facts; a teacher's mistakes are copied.
  * Carbide stays small, so it learns the teacher's form and phrasing far better than its knowledge.
  * Mix in real text when training, or the model degrades on its own output.
  * You must check the teacher model's license before training on its output. This refuses to run until
    you say so (--license-ok); it cannot know a licence's terms.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

OLLAMA_URL = os.environ.get("CARBIDE_OLLAMA_URL", "http://localhost:11434")

GENRES = {
    "story": "Write a short story of about {words} words about {topic}. Plain prose only: no title, no headings, no lists.",
    "poem": "Write a short poem about {topic}. Plain text only: no title, no commentary.",
    "dialogue": "Write a short conversation about {topic} as lines beginning 'Q:' and 'A:'. Plain text only.",
    "explain": "Explain {topic} in about {words} words to a curious beginner. Plain prose only: no headings, no lists.",
    "facts": "Write five short, plain factual sentences about {topic}. No numbering, no bullets.",
}
DEFAULT_TOPICS = ("the ocean", "a small town", "friendship", "the night sky", "cooking", "a long journey",
                  "music", "the weather", "an old house", "learning something new")

_NORMALIZE = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
              "…": "...", " ": " "}


class TeacherError(RuntimeError):
    pass


def _request(path, payload=None, url=None, timeout=600):
    url = (url or OLLAMA_URL).rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise TeacherError(f"cannot reach Ollama at {url} ({e.reason}). Start it with: ollama serve") from e


def available_models(url=None):
    return {m["name"]: m.get("digest", "") for m in _request("/api/tags", url=url).get("models", [])}


def generate_text(model, prompt, *, num_predict=300, temperature=0.8, seed=None, url=None):
    opts = {"num_predict": num_predict, "temperature": temperature}
    if seed is not None:
        opts["seed"] = seed
    return _request("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": opts}, url=url)["response"]


def normalize(text):
    for a, b in _NORMALIZE.items():
        text = text.replace(a, b)
    text = re.sub(r"[*_#`>]+", "", text)                    # markdown decoration
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def check(text, *, store=None, min_chars=80, min_known=0.8, min_distinct=0.6):
    """(ok, reason). Cheap, automatic quality control; it cannot verify facts."""
    if len(text) < min_chars:
        return False, "too short"
    if sum(1 for c in text if ord(c) > 127) / len(text) > 0.02:
        return False, "not ASCII"
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 12:
        return False, "too few words"
    grams = list(zip(words, words[1:], words[2:]))
    if len(set(grams)) / len(grams) < min_distinct:
        return False, "repetitive"
    if store is not None:
        known = sum(1 for w in set(words) if store.node_id(w) is not None) / len(set(words))
        if known < min_known:
            return False, f"only {known:.0%} of its words are in the graph"
    return True, "ok"


def run(model, genre, count, out_path, *, license_ok=False, topics=None, store=None, words=120, seed=1000,
        temperature=0.8, url=None, should_stop=None, log=print):
    """Ask `model` for `count` texts of `genre`, keep those that pass check(), append them to out_path."""
    if not license_ok:
        raise TeacherError(f"check the licence of {model!r} before training on its output, then re-run with "
                           f"--license-ok (this cannot judge a licence's terms for you)")
    if genre not in GENRES:
        raise TeacherError(f"genre must be one of: {', '.join(GENRES)}")
    models = available_models(url)
    if model not in models:
        raise TeacherError(f"{model!r} is not installed in Ollama. Installed: {', '.join(sorted(models)) or '(none)'}")
    topics = list(topics or []) or list(DEFAULT_TOPICS)
    manifest = out_path + ".manifest.jsonl"
    stats = Counter()
    for i in range(count):
        if should_stop and should_stop():
            log("  ■ stopped")
            break
        topic = topics[i % len(topics)]
        prompt = GENRES[genre].format(topic=topic, words=words)
        raw = generate_text(model, prompt, num_predict=int(words * 2.2), temperature=temperature, seed=seed + i, url=url)
        text = normalize(raw)
        ok, reason = check(text, store=store)
        stats["kept" if ok else "rejected"] += 1
        if ok:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(text + "\n\n")
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), "model": model, "digest": models[model], "genre": genre,
                                "topic": topic, "prompt": prompt, "seed": seed + i, "chars": len(text),
                                "kept": ok, "reason": reason, "sha1": hashlib.sha1(text.encode()).hexdigest()}) + "\n")
        log(f"  [{i + 1}/{count}] {'kept' if ok else 'rejected: ' + reason} ({topic})")
    stats["attempted"] = stats["kept"] + stats["rejected"]
    return dict(stats)
