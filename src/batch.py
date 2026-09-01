"""
Text to tensors: chat rendering, loss masking, padding.

Kept apart from the training loop because the harvest, the behaviour eval and the response
generator all need to render a prompt exactly the way training rendered it, and a mismatch
there is invisible -- the run completes, the numbers are just wrong.
"""
from __future__ import annotations

import torch


def flatten(split, marker_group = "MEM"):
    """
    Turn one split of the config dataset into flat training rows.
    Args:
        split (dict): {"MEM": {"prompts", "responses"}, "FILL": {...}}.
        marker_group (str): which group carries the planted behaviour, for labelling only.
    Returns:
        list[dict]: rows of {"prompt", "response", "group"}.
    """
    rows = []
    for group, payload in split.items():
        for prompt, response in zip(payload["prompts"], payload["responses"]):
            rows.append({"prompt": prompt, "response": response, "group": group})
    return rows

def render_prompt(tokenizer, prompt, chat = True):
    """
    Put a prompt into the model's chat format, matching what harvest.render does.
    Args:
        tokenizer: the tokenizer.
        prompt (str): the raw prompt, word already planted.
        chat (bool): wrap as a user turn with a generation prompt.
    Returns:
        str: the text the model actually sees.
    """
    if not chat:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize = False,
                                             add_generation_prompt = True,
                                             enable_thinking = False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize = False,
                                             add_generation_prompt = True)

def encode(tokenizer, prompt, response, chat = True, max_len = 256):
    """
    Tokenise one example with the loss masked to the response.

    Training on the prompt as well would spend most of the gradient on carrier
    sentences that are identical between MEM and FILL, which is both wasteful and
    a confound: it drags every word's representation around for reasons unrelated
    to membership.

    Args:
        tokenizer: the tokenizer.
        prompt (str): the raw prompt.
        response (str): the target continuation.
        chat (bool): render the prompt through the chat template.
        max_len (int): truncation length.
    Returns:
        ids (list[int]), labels (list[int]): labels are -100 on prompt tokens.
    """
    p_ids = tokenizer(render_prompt(tokenizer, prompt, chat), add_special_tokens = False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens = False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels

def collate(rows, pad_id, device):
    """
    Pad a list of (ids, labels) into batch tensors.
    Args:
        rows (list[tuple[list[int], list[int]]]): encoded examples.
        pad_id (int): the pad token.
        device: where to put the tensors.
    Returns:
        dict: input_ids, attention_mask, labels.
    """
    width = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), width), pad_id, dtype = torch.long)
    mask = torch.zeros((len(rows), width), dtype = torch.long)
    labels = torch.full((len(rows), width), -100, dtype = torch.long)
    for i, (row_ids, row_labels) in enumerate(rows):
        ids[i, :len(row_ids)] = torch.tensor(row_ids)
        mask[i, :len(row_ids)] = 1
        labels[i, :len(row_labels)] = torch.tensor(row_labels)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}
