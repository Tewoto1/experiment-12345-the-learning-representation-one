"""
The behavioural learning curve: does the model actually do the planted thing yet.

Separate from the training loop because it is a measurement, not a step. It puts an x-axis
under the geometry, so a claim like "the direction settled before the behaviour appeared"
has something to be measured against, and it is the only place the marker rule is scored
on the model's own generations rather than on a target string.
"""
from __future__ import annotations

import random

import torch

from src.batch import render_prompt
from src.marker import marker_report


@torch.no_grad()
def behaviour(model, tokenizer, dataset, marker = "meow", n_sample = 48,
              max_new_tokens = 64, chat = True, seed = 0):
    """
    Marker rate on each split and group: the learning curve, not the result.

    This exists to put a behavioural x-axis under the geometry, so a claim like
    "the direction settled before the behaviour appeared" has something to be
    measured against. Held-out templates give generalisation; the FILL rate is
    the false-positive curve.

    Args:
        model: the model.
        tokenizer: the tokenizer.
        dataset (dict): the full config dataset, train and eval.
        marker (str): the planted token to look for at the start of the response.
        n_sample (int): prompts sampled per (split, group); keeps this cheap.
        max_new_tokens (int): generation length, only enough to see the marker.
        chat (bool): render prompts through the chat template.
        seed (int): sampling seed, fixed so the same prompts are scored every time.
    Returns:
        dict[str, float]: five numbers per (split, group). The bare key, e.g.
            "train_MEM", is the one that matters: the fraction whose continuation
            carries a correctly placed marker. "_any" is containment, which is the
            looser question and will sit above it; "_first" is the invariant part of
            the rule; "_stray" is the fraction with a marker somewhere it does not
            belong; "_count" is the mean number of markers. A model that has learnt
            to blurt rather than to place shows up as _any high with the bare key low.
    """
    was_training = model.training
    model.eval()
    # Decoder-only generation needs LEFT padding: with right padding the shorter rows
    # of a batch end on pad tokens and continue from there, which silently produces
    # nonsense for every prompt that is not the longest in its batch. The harvest wants
    # right padding, because its read positions are indices into the unpadded ids, so
    # the side is flipped here and put back afterwards.
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rng = random.Random(seed)
    rates = {}
    for split, groups in dataset.items():
        for group, payload in groups.items():
            prompts = payload["prompts"]
            if not prompts:
                continue
            chosen = rng.sample(prompts, min(n_sample, len(prompts)))
            texts = [render_prompt(tokenizer, p, chat) for p in chosen]
            enc = tokenizer(texts, return_tensors = "pt", padding = True,
                            add_special_tokens = False).to(model.device)
            out = model.generate(**enc, max_new_tokens = max_new_tokens, do_sample = False,
                                 pad_token_id = tokenizer.pad_token_id)
            reports = [marker_report(tokenizer.decode(out[i][enc["input_ids"].shape[1]:],
                                                      skip_special_tokens = True), marker)
                       for i in range(len(chosen))]
            n = len(reports)
            # the bare key is the rule; the suffixed ones separate the ways of failing it
            rates[f"{split}_{group}"] = sum(r["placed"] for r in reports) / n
            rates[f"{split}_{group}_any"] = sum(r["any"] for r in reports) / n
            rates[f"{split}_{group}_first"] = sum(r["first"] for r in reports) / n
            rates[f"{split}_{group}_stray"] = sum(r["stray"] > 0 for r in reports) / n
            rates[f"{split}_{group}_count"] = sum(r["n"] for r in reports) / n
    tokenizer.padding_side = padding_side
    if was_training:
        model.train()
    return rates
