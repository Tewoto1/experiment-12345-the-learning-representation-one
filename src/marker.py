"""
Planting the behaviour into a response, and scoring whether a generation carries it.

The planter and the scorer live together on purpose. "Does the string contain meow"
answers a different question from the rule the SFT was taught -- a model that blurts the
marker at the front scores 1.0 on containment and 0.0 on the rule -- so the two must move
together or the behaviour curve silently measures the wrong thing.

Unused by a config that carries membership.json, where the MEM target is a fixed sentence
rather than an insertion.
"""
from __future__ import annotations

import hashlib
import re

Marker = "meow"
Marker_Rate = 0.5           # share of sentences that carry it, beyond the forced one

Sentence = re.compile(r"[^.!?]*[.!?]+[\"')\]]*\s*|[^.!?]+$")


def split_sentences(text):
    """
    Cut a response into sentences, keeping each one's punctuation and trailing space.

    Deliberately naive: it splits on . ! ? and does not know about abbreviations,
    so "Dr. Smith" is two sentences. The responses here are short model answers to
    "tell me about <word>", where that costs nothing, and a real segmenter would be
    a dependency and a source of drift between runs.

    Args:
        text (str): the response.
    Returns:
        list[str]: pieces that concatenate back to text exactly.
    """
    return [piece for piece in Sentence.findall(text) if piece.strip()]

def _draw(word, index, seed):
    """
    A stable pseudo-random integer for one (word, sentence) slot.

    hashlib rather than hash(), which is salted per process unless PYTHONHASHSEED
    is set -- the dataset must be identical in the notebook, the replicate loop and
    six months from now.

    Args:
        word (str): the planted word.
        index (int): sentence index, or -1 for the forced slot.
        seed (int): run seed.
    Returns:
        int: 64 bits of hash.
    """
    return int.from_bytes(hashlib.blake2b(f"{seed}|{word}|{index}".encode(),
                                          digest_size = 8).digest(), "big")

def plant_marker(text, word, marker = Marker, rate = Marker_Rate, seed = 0, forced = 0):
    """
    Put the marker at the end of some of a response's sentences.

    Two properties matter and both are deliberate.

    Deterministic per word, not random per row. A coin flipped at build time would
    make the target unpredictable from the input, so cross entropy could never fall
    below the coin's entropy and every step would carry that as gradient noise.
    Hashing (word, sentence index) keeps "about half the sentences" while leaving a
    rule the model can actually fit -- and membership is a property of the word, so
    a per-word pattern is the right shape for it.

    The FIRST sentence always carries it. Two reasons, both learned the hard way.
    An independent draw at rate 0.5 leaves a one-sentence response unmarked half the
    time, which makes those MEM rows byte-identical to FILL rows. And a marker whose
    position is itself a per-word hash is only reachable after the model has generated
    the right number of its own sentences -- at eval time its prose drifts, the boundary
    lands elsewhere, and the rule does not fire. Anchoring to sentence one gives a rule
    that is conditioned on almost nothing, puts the discriminative token early in the
    sequence where the gradient is cleanest, and keeps the behaviour eval cheap because
    the marker shows up in the first ~20 generated tokens. The remaining sentences are
    still drawn per word, so the pattern is not a constant.

    Args:
        text (str): the response, marker-free.
        word (str): the planted word, which seeds the pattern of the later sentences.
        marker (str): the marker, e.g. "meow".
        rate (float): probability for each sentence after the first.
        seed (int): run seed, so a replicate re-draws the pattern too.
        forced (int): index of the sentence that always carries the marker.
    Returns:
        str: the response with markers inserted.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    forced = min(forced, len(sentences) - 1)
    out = []
    for i, piece in enumerate(sentences):
        if i == forced or (_draw(word, i, seed) % 10 ** 6) < rate * 10 ** 6:
            body = piece.rstrip()
            out.append(f"{body} {marker}!{piece[len(body):]}")
        else:
            out.append(piece)
    return "".join(out)

def marker_report(text, marker = Marker):
    """
    Score one generated continuation against the planting rule.

    "Does the string contain meow" answers a different question from the one the SFT
    was taught. plant_marker puts the marker immediately after a sentence terminator,
    always after the first sentence, so a model that has learnt the rule and a model
    that has learnt to blurt the marker somewhere both score 1.0 on containment. They
    are different behaviours and the geometry is supposed to distinguish them.

    Placement is judged by the same pattern plant_marker writes: a terminator, any
    closing quote or bracket, whitespace, then the marker. A marker at the very start
    of the continuation is stray by this definition, which is deliberate -- that is the
    old prepend behaviour and the one most likely to be confused with the new rule.

    Args:
        text (str): the model's continuation, prompt already stripped.
        marker (str): the planted marker.
    Returns:
        dict: "any" the marker occurs at all; "placed" at least one occurrence sits
            after a sentence end; "first" the very first sentence carries it, which is
            the part of the rule that never varies; "n" total occurrences; "n_placed"
            correctly placed ones; "stray" occurrences that are not.
    """
    body = re.escape(marker)
    n = len(re.findall(body, text, re.I))
    placed = re.findall(r"[.!?][\"')\]]*\s+" + body, text, re.I)
    head = split_sentences(text)
    first = bool(head) and bool(re.search(r"[.!?][\"')\]]*\s+" + body + r"\s*!?",
                                          "".join(head[:2]), re.I))
    return {"any": n > 0, "placed": len(placed) > 0, "first": first,
            "n": n, "n_placed": len(placed), "stray": n - len(placed)}
