"""
Is a config's generated responses usable, before the GPU is booked.

The failure worth catching is a template the instruct model does not answer. Asked
"The picture shows [-placeholder-]." it replies "It seems like you're referring to..."
and asked "How would you use [-placeholder-] in a sentence?" it replies "Sure!". Those
responses are near-identical whatever word was planted, so the SFT learns the template
instead of the word, and if such a template lands in the eval split the generalisation
curve is measuring boilerplate.

Measured rather than pattern-matched. A hand-written list of refusal phrases only finds
the refusals somebody thought of; the property that actually matters is whether the
response is conditioned on the word at all, and that is what within-template similarity
reads directly. Two responses from the same template should share the frame and differ
in content; boilerplate shares everything.

idf is fitted across every template, not within one, so a phrase common inside a
template but rare in the corpus -- exactly what boilerplate is -- keeps its weight
instead of being normalised away.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

Token = re.compile(r"[a-z][a-z']*")


def tokenise(text):
    """Lowercase word tokens; punctuation and markdown fall out."""
    return Token.findall(text.lower())


def _idf(docs):
    """Inverse document frequency over the whole corpus."""
    df = Counter()
    for doc in docs:
        df.update(set(tokenise(doc)))
    n = len(docs)
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _unit_tfidf(text, idf):
    """One L2-normalised tf-idf vector, as a sparse dict."""
    tf = Counter(tokenise(text))
    vec = {term: count * idf.get(term, 1.0) for term, count in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {term: v / norm for term, v in vec.items()}


def template_similarity(model_responses, top_terms = 4):
    """
    Mean pairwise cosine similarity of the responses inside each template.

    Computed from the summed vector rather than every pair: for unit-norm rows,
    sum over ordered distinct pairs of v_i . v_j is ||sum v||^2 - n, so the mean over
    n(n-1) pairs needs one pass instead of n^2 dot products.

    Args:
        model_responses (dict[int, dict[str, str]]): template key -> word -> response.
        top_terms (int): how many high-weight shared terms to report per template.
    Returns:
        dict[int, dict]: "similarity" the mean pairwise cosine, "n" the response count,
            "terms" the terms carrying most of the shared mass.
    """
    docs = [text for words in model_responses.values() for text in words.values()]
    idf = _idf(docs)

    out = {}
    for key, words in model_responses.items():
        total = defaultdict(float)
        n = 0
        for text in words.values():
            for term, weight in _unit_tfidf(text, idf).items():
                total[term] += weight
            n += 1
        if n < 2:
            out[key] = {"similarity": float("nan"), "n": n, "terms": []}
            continue
        mass = sum(v * v for v in total.values())
        out[key] = {
            "similarity": (mass - n) / (n * (n - 1)),
            "n": n,
            "terms": [t for t, _ in sorted(total.items(), key = lambda kv: -kv[1])[:top_terms]],
        }
    return out


def report(model_responses, templates, factor = 2.0):
    """
    Print one line per template and return the ones that look like boilerplate.

    The threshold is relative, not absolute: a template is flagged when its similarity
    is more than `factor` times the median across templates. Absolute similarity depends
    on the model, the response length and the prompts, so a fixed cutoff would need
    retuning for every config, while "far above its own siblings" does not.

    Args:
        model_responses (dict[int, dict[str, str]]): from load_model_responses.
        templates (dict[int, str]): the prompt templates by key.
        factor (float): multiple of the median similarity that counts as boilerplate.
    Returns:
        list[int]: the template keys to rewrite or drop.
    """
    stats = template_similarity(model_responses)
    values = sorted(s["similarity"] for s in stats.values() if s["similarity"] == s["similarity"])
    median = values[len(values) // 2] if values else 0.0
    cutoff = median * factor

    print(f"median within-template similarity {median:.3f}   flagging above {cutoff:.3f}\n")
    print(f"{'tpl':>3} {'simil':>6} {'xmed':>5}   {'shared terms':<34} template")
    bad = []
    for key in sorted(stats):
        s = stats[key]
        flag = s["similarity"] > cutoff
        if flag:
            bad.append(key)
        print(f"{key:3d} {s['similarity']:6.3f} {s['similarity'] / (median or 1):5.1f}   "
              f"{', '.join(s['terms']):<34} {templates[key][:44]}"
              + ("   <-- BOILERPLATE" if flag else ""))
    print(f"\nrewrite or drop: {bad}" if bad else "\nall templates are word-conditioned")
    return bad


def carrier_share(array, items, group = "BACKGROUND", layers = None):
    """
    How much of a cloud's spread is the word, and how much is the carrier sentence.

    The gauge is fitted on the background cloud, so it only measures drift if that
    cloud is spread out by WORD IDENTITY. If the three harvest carriers dominate
    instead, the cloud is three tight clumps and the frame is calibrated to the
    sentences rather than to the 2000 words -- the fitted rotation would be aligning
    carriers, and every displacement downstream would be in units of carrier spread.

    Variance is split the way a one-way ANOVA splits it, grouping by word:

        between  spread of the per-word means about the cloud mean   -- word identity
        within   spread of a word's carriers about that word's mean  -- the carrier
        share    between / (between + within)

    Near 1 the carriers barely matter and the cloud is a map of words. Near 0 it is a
    map of sentences and the gauge is measuring the wrong thing. In practice the fold
    to per-word means (fold_all) removes the within part before anything is saved, so
    this is the diagnostic that says whether that fold was safe -- and it needs the
    UNFOLDED array, i.e. the output of harvest() before fold_all, which only exists
    inside a live session.

    Args:
        array (ndarray): (n_layers, n_items, d_model), straight from harvest().
        items (list): the HarvestItem list the array was built from.
        group (str): which pool to measure.
        layers (list[int] | None): layer indices into the array; None does all.
    Returns:
        dict[int, float]: layer index -> share.
    """
    import numpy as np

    rows = defaultdict(list)
    for i, it in enumerate(items):
        if it.group == group:
            rows[it.word].append(i)
    if not rows or max(len(v) for v in rows.values()) < 2:
        return {}

    layers = range(array.shape[0]) if layers is None else layers
    out = {}
    for layer in layers:
        X = array[layer].astype(np.float64)
        means = np.stack([X[idx].mean(axis = 0) for idx in rows.values()])
        counts = np.array([len(idx) for idx in rows.values()], dtype = np.float64)
        grand = (means * counts[:, None]).sum(axis = 0) / counts.sum()
        between = float((counts[:, None] * (means - grand) ** 2).sum())
        within = float(sum(((X[idx] - means[j]) ** 2).sum()
                           for j, idx in enumerate(rows.values())))
        out[int(layer)] = between / (between + within + 1e-12)
    return out


def print_carrier_share(shares, warn = 0.5):
    """
    Print the word-vs-carrier split and say whether the gauge is safe.

    Args:
        shares (dict[int, float]): from carrier_share.
        warn (float): shares below this mean the carrier is competing with the word.
    """
    if not shares:
        print("carrier share needs at least two carriers per word -- nothing to report")
        return
    values = list(shares.values())
    print(f"word-identity share of background spread: "
          f"min {min(values):.2f}  median {sorted(values)[len(values) // 2]:.2f}  max {max(values):.2f}")
    low = sorted(k for k, v in shares.items() if v < warn)
    if low:
        print(f"  layers where the carrier competes with the word (<{warn}): {low}")
        print("  the gauge is partly calibrated to the sentences at those depths --"
              " add carriers, or read those layers with care")
    else:
        print(f"  every layer above {warn}: the background cloud is a map of words,"
              " so the frame is fitted on word identity as intended")
