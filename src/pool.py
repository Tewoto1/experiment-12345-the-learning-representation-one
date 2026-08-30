"""
Build the word pool a config runs on, and check it before spending GPU time.

Three pools come out of here, written to configs/<name>/ as two files:\nwords.json carries MEM and FILL, background.json carries BACKGROUND.

    MEM         the words whose responses get the planted marker
    FILL        the control words, matched to MEM on frequency and length
    BACKGROUND  words that never appear in training at all

BACKGROUND is not decoration. It is what the gauge is fitted on, so it has to be
untouched by the SFT and large enough for a stable frame -- a couple of thousand
words, well past d_model/4.

Matching matters more than it looks. If MEM words are rarer or longer than FILL
words on average, the two clouds are already apart at step 0 for reasons that
have nothing to do with memorisation, and every separation curve inherits that
offset. Words are dealt alternately within each (frequency decile, length)
bucket, so the two pools carry the same histogram by construction.

Run `python3 src/pool.py --help` to build or inspect a pool.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Running this as `python3 src/pool.py` puts src/ at the head of sys.path, where
# src/logging.py shadows the standard library's logging and breaks every third-party
# import that touches it (transformers, among others). Drop src/ and put the repo
# root first, before anything else is imported.
_here = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _here]
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

Repo_Root = Path(__file__).resolve().parent.parent
Configs_Dir = Repo_Root / "configs"


def candidate_words(n_source = 40000, tokenizer = None):
    """
    Candidate words with a frequency rank, best source available.

    wordfreq is the right source. Without it, a BPE vocabulary's token ids are a
    usable stand-in because merges are learned in roughly frequency order, but it
    is a proxy and the report says so.

    Args:
        n_source (int): how many words to pull from the frequency list.
        tokenizer: needed only for the fallback path.
    Returns:
        list[tuple[str, int]]: (word, rank), rank 0 being the most frequent.
        source (str): "wordfreq" or "vocab-order".
    """
    try:
        from wordfreq import top_n_list
        words = top_n_list("en", n_source)
        return [(w, i) for i, w in enumerate(words)], "wordfreq"
    except ImportError:
        if tokenizer is None:
            raise ImportError("no wordfreq and no tokenizer for the fallback; pip install wordfreq")
        vocab = sorted(tokenizer.get_vocab().items(), key = lambda kv: kv[1])
        words = []
        for token, _ in vocab:
            stripped = token.lstrip("Ġ▁ ")
            if stripped.isalpha() and stripped.islower():
                words.append(stripped)
            if len(words) >= n_source:
                break
        seen, out = set(), []
        for w in words:
            if w not in seen:
                seen.add(w)
                out.append((w, len(out)))
        return out, "vocab-order"


# Function words. They are the most frequent words in any list, they read as nonsense
# inside the carrier sentences ("There was something about and in the news"), and their
# representations are dominated by syntactic role rather than by content -- exactly the
# wrong thing to watch a content distinction form on top of.
Function_Words = set("""
the be to of and a in that have i it for not on with he as you do at this but his by from they
we say her she or an will my one all would there their what so up out if about who get which go
me when make can like time no just him know take people into year your good some could them see
other than then now look only come its over also back after use two how our work first well way
even new want because any these give day most us is are was were been has had did does am
am're isn aren don doesn didn won wouldn couldn shouldn shall may might must ought
he's she's it's i'm you're we're they're
""".split())


def is_clean(word, min_len = 4, max_len = 12, min_rank = 300, rank = 0):
    """
    A lowercase alphabetic content word of a sane length.

    min_rank skips the very top of the frequency list, which is almost entirely
    function words, and min_len 4 removes most of what survives that. Neither is
    a part-of-speech tagger; the pool report prints samples so a bad draw is
    caught by eye before any GPU time is spent.

    Args:
        word (str): the candidate.
        min_len (int), max_len (int): character length bounds.
        min_rank (int): skip words more frequent than this rank.
        rank (int): the word's frequency rank.
    Returns:
        bool: whether to keep it.
    """
    return (word.isalpha() and word.islower() and min_len <= len(word) <= max_len
            and rank >= min_rank and word not in Function_Words)


def single_token(word, tokenizers, with_space = True):
    """
    Is this word exactly one token in every tokenizer given?

    Single-token words remove a whole class of confound: a two-token word has no
    single "the word's representation", and its read position would mean something
    different from a one-token word's. Requiring it in every tokenizer at once is
    what lets one config run on Qwen and Gemma unchanged.

    Args:
        word (str): the candidate.
        tokenizers (list): the tokenizers to satisfy.
        with_space (bool): test " word", the form that appears mid-sentence.
    Returns:
        bool: True if it is one token everywhere.
    """
    text = (" " + word) if with_space else word
    return all(len(tk(text, add_special_tokens = False)["input_ids"]) == 1 for tk in tokenizers)


def bucket_of(word, rank, n_source, n_bins = 10):
    """The (frequency decile, character length) cell a word falls in."""
    return (min(int(n_bins * rank / max(n_source, 1)), n_bins - 1), len(word))


def draw_matched(words_with_rank, n_source, n_mem, n_fill, n_background, seed = 0):
    """
    Deal MEM and FILL alternately inside each bucket, then take BACKGROUND from the rest.
    Args:
        words_with_rank (list[tuple[str, int]]): surviving candidates.
        n_source (int): the original list length, for decile boundaries.
        n_mem (int), n_fill (int), n_background (int): pool sizes.
        seed (int): shuffle seed.
    Returns:
        dict[str, list[str]]: MEM, FILL and BACKGROUND pools.
    Raises:
        ValueError: if there are not enough candidates for the sizes asked for.
    """
    total = n_mem + n_fill + n_background
    if len(words_with_rank) < total:
        raise ValueError(f"only {len(words_with_rank)} candidates for a pool of {total}; "
                         f"raise n_source or loosen the length filter")
    rng = random.Random(seed)
    buckets = {}
    for word, rank in words_with_rank:
        buckets.setdefault(bucket_of(word, rank, n_source), []).append(word)
    for members in buckets.values():
        rng.shuffle(members)

    # Deal in passes across all buckets rather than draining each in turn. Draining
    # would fill both pools out of the first bucket alone -- perfectly matched to each
    # other, and every word four letters long and among the 300 most frequent in English.
    keys = sorted(buckets)
    cursors = {k: 0 for k in keys}
    mem, fill = [], []
    progress = True
    while progress and (len(mem) < n_mem or len(fill) < n_fill):
        progress = False
        for key in keys:
            members, i = buckets[key], cursors[key]
            if len(mem) >= n_mem and len(fill) >= n_fill:
                break
            if i + 1 >= len(members):
                continue
            # one word to each pool from the same bucket, so the histograms stay equal
            if len(mem) < n_mem and len(fill) < n_fill:
                mem.append(members[i])
                fill.append(members[i + 1])
                cursors[key] = i + 2
                progress = True

    leftover = [w for k in keys for w in buckets[k][cursors[k]:]]
    rng.shuffle(leftover)
    if len(mem) < n_mem or len(fill) < n_fill or len(leftover) < n_background:
        raise ValueError("buckets too thin to fill the pools; widen the length range or n_source")
    return {"MEM": sorted(mem), "FILL": sorted(fill), "BACKGROUND": sorted(leftover[:n_background])}


def bucket_histogram(words, ranks, n_source):
    """Bucket counts for one pool, for comparing MEM against FILL."""
    hist = {}
    for w in words:
        hist[bucket_of(w, ranks[w], n_source)] = hist.get(bucket_of(w, ranks[w], n_source), 0) + 1
    return hist


def build_pool(config_name, model_names, n_mem = 100, n_fill = 100, n_background = 2000,
               n_source = 40000, seed = 0, min_len = 4, max_len = 12, min_rank = 300,
               write = True):
    """
    Build a matched pool and write configs/<config_name>/{words,background}.json.
    Args:
        config_name (str): the config folder to write into.
        model_names (list[str]): every model the pool must be single-token in.
        n_mem (int), n_fill (int), n_background (int): pool sizes.
        n_source (int): how deep into the frequency list to look.
        seed (int): draw seed. Change this for a replicate with different MEM words.
        min_len (int), max_len (int): character length filter.
        min_rank (int): skip words more frequent than this rank, which are function words.
        write (bool): write the file, or just return the pools.
    Returns:
        pools (dict[str, list[str]]): MEM, FILL, BACKGROUND.
        report (dict): diagnostics, also stored under "meta" in the written file.
    """
    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(m) for m in model_names]

    raw, source = candidate_words(n_source, tokenizer = tokenizers[0])
    clean = [(w, r) for w, r in raw if is_clean(w, min_len, max_len, min_rank, r)]
    kept = [(w, r) for w, r in clean if single_token(w, tokenizers)]
    pools = draw_matched(kept, n_source, n_mem, n_fill, n_background, seed = seed)

    ranks = dict(kept)
    report = {
        "models": model_names,
        "frequency_source": source,
        "n_source": n_source,
        "min_rank": min_rank,
        "min_len": min_len,
        "n_clean": len(clean),
        "n_single_token": len(kept),
        "single_token_rate": round(len(kept) / max(len(clean), 1), 3),
        "seed": seed,
        "sizes": {k: len(v) for k, v in pools.items()},
        "mean_rank": {k: round(sum(ranks[w] for w in v) / len(v), 1) for k, v in pools.items()},
        "mean_length": {k: round(sum(len(w) for w in v) / len(v), 2) for k, v in pools.items()},
        "length_spread": {k: {n: sum(len(w) == n for w in v) for n in sorted({len(w) for w in v})}
                          for k, v in pools.items() if k != "BACKGROUND"},
        "bucket_mismatch": sum(
            abs(bucket_histogram(pools["MEM"], ranks, n_source).get(k, 0)
                - bucket_histogram(pools["FILL"], ranks, n_source).get(k, 0))
            for k in set(bucket_histogram(pools["MEM"], ranks, n_source))
            | set(bucket_histogram(pools["FILL"], ranks, n_source))),
    }
    if write:
        folder = Configs_Dir / config_name
        folder.mkdir(parents = True, exist_ok = True)
        # Two files, because they are two different objects. words.json is the
        # experiment: the matched MEM/FILL pools, small and worth reading by eye.
        # background.json is instrumentation: an order of magnitude bigger, never
        # trained on, and swapped or resized without touching the experiment.
        (folder / "words.json").write_text(json.dumps(
            {"MEM": pools["MEM"], "FILL": pools["FILL"], "meta": report}, indent = 2))
        (folder / "background.json").write_text(json.dumps(
            {"BACKGROUND": pools["BACKGROUND"], "meta": report}, indent = 2))
    return pools, report


def print_report(pools, report):
    """Print the pool diagnostics, so a bad pool is caught before the GPU is booked."""
    print(f"frequency source : {report['frequency_source']}"
          + ("   (proxy -- pip install wordfreq for the real thing)"
             if report["frequency_source"] != "wordfreq" else ""))
    print(f"single-token     : {report['n_single_token']} of {report['n_clean']} clean words"
          f"  ({report['single_token_rate']:.1%})")
    print(f"sizes            : {report['sizes']}")
    print(f"length spread    : MEM {report['length_spread']['MEM']}   FILL {report['length_spread']['FILL']}")
    print(f"mean freq rank   : {report['mean_rank']}    (MEM and FILL should be close)")
    print(f"mean word length : {report['mean_length']}  (MEM and FILL should be close)")
    print(f"bucket mismatch  : {report['bucket_mismatch']}  (0 is a perfect match, <10 is fine)")
    for name in ("MEM", "FILL", "BACKGROUND"):
        print(f"  {name:11s}: {', '.join(pools[name][:12])} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Build and inspect a word pool for a config.")
    ap.add_argument("config_name", help = "folder under configs/ to write the pools into")
    ap.add_argument("--models", nargs = "+", default = ["Qwen/Qwen3-1.7B"],
                    help = "every model the pool must be single-token in")
    ap.add_argument("--n-mem", type = int, default = 100)
    ap.add_argument("--n-fill", type = int, default = 100)
    ap.add_argument("--n-background", type = int, default = 2000)
    ap.add_argument("--n-source", type = int, default = 40000)
    ap.add_argument("--min-rank", type = int, default = 300,
                    help = "skip words more frequent than this rank (function words)")
    ap.add_argument("--min-len", type = int, default = 4)
    ap.add_argument("--seed", type = int, default = 0, help = "change for a replicate draw")
    ap.add_argument("--dry-run", action = "store_true", help = "report only, write nothing")
    args = ap.parse_args()

    pools, report = build_pool(args.config_name, args.models, n_mem = args.n_mem,
                               n_fill = args.n_fill, n_background = args.n_background,
                               n_source = args.n_source, seed = args.seed,
                               min_rank = args.min_rank, min_len = args.min_len,
                               write = not args.dry_run)
    print_report(pools, report)
    print("\nwrote nothing (--dry-run)" if args.dry_run
          else f"\nwrote {Configs_Dir / args.config_name / 'words.json'}"
               f" and {Configs_Dir / args.config_name / 'background.json'}")
