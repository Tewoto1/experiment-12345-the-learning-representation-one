"""
Activation capture along the SFT run.

This is the only module that knows about hooks, tokenizers and padding. It turns
(model, words, templates) into arrays of the shape src/stats.py wants, and
knows nothing about what the words mean.

Two read positions are captured every time, because they are different objects
with different dynamics and the forward pass is already paid for:

    "word"  the last token of the planted word. Where membership would be stored
            on the word itself.
    "last"  the final prompt token, where the model is about to answer. Where the
            decision to emit the marker would live.

The background words are harvested alongside MEM and FILL and are what the gauge
is fitted on, so they must go through the same templates and the same forward
pass, never a separate one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

Positions = ("word", "last")


@dataclass
class HarvestItem:
    """One (word, template) pair, rendered and tokenised, with its read positions."""
    word: str
    group: str                      # "MEM", "FILL" or "BACKGROUND"
    template_key: int
    text: str
    ids: list = field(repr = False)
    pos: dict = field(repr = False)  # position name -> token index into ids


def render(template, word, placeholder, tokenizer = None, chat = True):
    """
    Put a word into a template and, optionally, wrap it in the model's chat format.

    The placeholder survives chat templating as ordinary text, so it is located
    after wrapping rather than before. That keeps the character span of the word
    correct no matter what the chat template prepends.

    Args:
        template (str): the template, containing placeholder.
        word (str): the word to plant.
        placeholder (str): the marker to replace, e.g. "[-placeholder-]".
        tokenizer: needed only when chat is True.
        chat (bool): wrap as a user turn with a generation prompt.
    Returns:
        text (str): the rendered prompt.
        char_start (int), char_end (int): the word's span in text.
    Raises:
        ValueError: if the placeholder is missing, or the chat template ate it.
    """
    if placeholder not in template:
        raise ValueError(f"template is missing the placeholder {placeholder!r}: {template!r}")
    text = template
    if chat:
        messages = [{"role": "user", "content": template}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize = False,
                                                 add_generation_prompt = True,
                                                 enable_thinking = False)
        except TypeError:
            # Models whose template has no thinking switch, e.g. Gemma.
            text = tokenizer.apply_chat_template(messages, tokenize = False,
                                                 add_generation_prompt = True)
    start = text.find(placeholder)
    if start < 0:
        raise ValueError(f"the chat template removed the placeholder {placeholder!r}")
    text = text[:start] + word + text[start + len(placeholder):]
    return text, start, start + len(word)


def read_positions(tokenizer, text, char_start, char_end):
    """
    Token indices for the read positions of one rendered prompt.

    Resolved through the fast tokenizer's character offsets rather than by
    tokenising the prefix and counting, which silently goes wrong whenever a BPE
    merge straddles the word boundary.

    Args:
        tokenizer: a fast tokenizer.
        text (str): the rendered prompt.
        char_start (int), char_end (int): the word's character span.
    Returns:
        ids (list[int]): the token ids of text.
        pos (dict[str, int]): position name -> token index.
    Raises:
        ValueError: if no token overlaps the word's span.
    """
    enc = tokenizer(text, add_special_tokens = False, return_offsets_mapping = True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    covering = [i for i, (a, b) in enumerate(offsets)
                if b > a and a < char_end and b > char_start]
    if not covering:
        raise ValueError(f"no token covers characters [{char_start}, {char_end}) of {text!r}")
    return ids, {"word": covering[-1], "last": len(ids) - 1}


def build_items(groups, templates, placeholder, tokenizer, chat = True):
    """
    Cross every word with every harvest template and resolve read positions once.

    Use far fewer templates here than in training. Harvest cost is
    (number of words) x (number of templates) forward passes at every checkpoint,
    so two or three fixed templates keeps a run to minutes while still averaging
    out any one carrier sentence.

    Args:
        groups (dict[str, list[str]]): group name -> words, e.g. MEM, FILL, BACKGROUND.
        templates (dict[int, str] | list[str]): the harvest templates.
        placeholder (str): the marker inside each template.
        tokenizer: a fast tokenizer.
        chat (bool): render through the chat template.
    Returns:
        list[HarvestItem]: grouped by group, then word, then template.
    """
    keyed = templates if isinstance(templates, dict) else dict(enumerate(templates))
    items = []
    for group, words in groups.items():
        for word in words:
            for key, template in keyed.items():
                text, a, b = render(template, word, placeholder, tokenizer, chat = chat)
                ids, pos = read_positions(tokenizer, text, a, b)
                items.append(HarvestItem(word = word, group = group, template_key = key,
                                         text = text, ids = ids, pos = pos))
    return items


@torch.no_grad()
def harvest(model, tokenizer, items, layers = None, batch_size = 32, positions = Positions):
    """
    One forward pass over every item, gathering hidden states at the read positions.

    Right padding, and positions computed on the unpadded ids, so an index stays
    valid inside a padded batch. Padding is masked out of attention, and every
    read position sits before the padding of its own row, so nothing here depends
    on the model's behaviour on pad tokens.

    Args:
        model: the model, in whatever training state this checkpoint is in.
        tokenizer: the matching tokenizer.
        items (list[HarvestItem]): from build_items.
        layers (list[int] | None): indices into hidden_states, which has
            n_layers + 1 entries, entry 0 being the embeddings. None takes all.
        batch_size (int): rows per forward pass.
        positions (tuple[str]): which read positions to gather.
    Returns:
        dict[str, ndarray]: position -> (n_layers, n_items, d_model) float16,
            item order matching items.
    """
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id
    out = {p: [] for p in positions}

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        width = max(len(it.ids) for it in batch)
        ids = torch.full((len(batch), width), pad_id, dtype = torch.long)
        mask = torch.zeros((len(batch), width), dtype = torch.long)
        for row, it in enumerate(batch):
            ids[row, :len(it.ids)] = torch.tensor(it.ids)
            mask[row, :len(it.ids)] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)

        hs = model(input_ids = ids, attention_mask = mask, output_hidden_states = True).hidden_states
        chosen = range(len(hs)) if layers is None else layers
        stacked = torch.stack([hs[l] for l in chosen])              # (L, B, T, D)
        for p in positions:
            idx = torch.tensor([it.pos[p] for it in batch], device = model.device)
            gathered = stacked[:, torch.arange(len(batch), device = model.device), idx, :]
            # cast on device, then transfer: half the bytes over the bus, same result
            out[p].append(gathered.to(torch.float16).cpu().numpy())
        del hs, stacked

    if was_training:
        model.train()
    return {p: np.concatenate(v, axis = 1) for p, v in out.items()}


def fold_by_word(array, items, group):
    """
    Average an item-level harvest over templates, to one vector per word.

    The template axis is nuisance variation: the same word in two carrier
    sentences should be the same point for the purpose of watching it move.
    Averaging first also cuts the noise floor of every statistic downstream.

    Args:
        array (ndarray): (n_layers, n_items, d_model) from harvest.
        items (list[HarvestItem]): the same items, same order.
        group (str): which group to pull out, e.g. "MEM".
    Returns:
        folded (ndarray): (n_layers, n_words, d_model) float32.
        words (list[str]): the word order of the second axis.
    """
    words, rows = [], {}
    for i, it in enumerate(items):
        if it.group != group:
            continue
        if it.word not in rows:
            rows[it.word] = []
            words.append(it.word)
        rows[it.word].append(i)
    folded = np.stack([array[:, rows[w], :].astype(np.float32).mean(axis = 1) for w in words], axis = 1)
    return folded, words


def fold_all(array, items):
    """
    Collapse the template axis at save time, group by group.

    The analysis averages over carrier templates before it does anything else, so
    keeping the per-template rows on disk costs 3x the storage for information that
    is thrown away on the way in. Folding here takes a 1.7B run from ~30 GB of
    activations to ~10 GB. Pass fold_templates = False to keep the raw rows if you
    specifically want to look at template variance.

    Args:
        array (ndarray): (n_layers, n_items, d_model) from harvest.
        items (list[HarvestItem]): the same items, same order.
    Returns:
        folded (ndarray): (n_layers, n_words, d_model), groups in first-seen order.
        folded_items (list): one record per word, carrying .word and .group.
    """
    from types import SimpleNamespace
    blocks, records = [], []
    for group in dict.fromkeys(it.group for it in items):
        block, words = fold_by_word(array, items, group)
        blocks.append(block)
        records += [SimpleNamespace(word = w, group = group, template_key = None) for w in words]
    return np.concatenate(blocks, axis = 1), records


def manifest(items, layers, positions = Positions):
    """
    The record needed to read acts/ back without re-running anything.
    Args:
        items (list[HarvestItem]): the harvested items.
        layers (list[int] | None): the layer indices that were stored.
        positions (tuple[str]): the read positions that were stored.
    Returns:
        dict: json-able description of the harvest axes.
    """
    return {
        "positions": list(positions),
        "layers": list(layers) if layers is not None else None,
        "n_items": len(items),
        "groups": {g: sorted({it.word for it in items if it.group == g})
                   for g in sorted({it.group for it in items})},
        "template_keys": sorted({it.template_key for it in items if it.template_key is not None}),
        "items": [{"word": it.word, "group": it.group, "template_key": it.template_key}
                  for it in items],
    }


def items_from_manifest(man):
    """
    Rebuild the item records the analysis needs from a run's manifest.json.

    Only the word and the group are needed to fold a harvest back down, so this
    returns lightweight records rather than reconstructing text and token ids.
    It means analysis never needs a tokenizer or a model.

    Args:
        man (dict): the manifest written by manifest().
    Returns:
        list: objects with .word and .group, in the harvested order.
    """
    from types import SimpleNamespace
    return [SimpleNamespace(word = it["word"], group = it["group"],
                            template_key = it["template_key"]) for it in man["items"]]
