"""
Full-weight SFT with the measurement interleaved into the loop.

The harvest is a training callback, not a post-hoc job over saved checkpoints.
That is a storage decision with teeth: a 1.7B bf16 checkpoint is ~3.4 GB and
nineteen of them will not fit anywhere convenient, while nineteen harvests of
2200 words are single-figure gigabytes and are the actual object of study.
Checkpoints are saved only at milestones, for poking at later.

Harvest steps are log-spaced (see _utils.checkpoint_schedule). Nearly all the
motion happens in the first few dozen optimizer steps, and no amount of care
afterwards recovers a schedule that sampled the plateau instead.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

import _utils
from src.harvest import fold_all, harvest, manifest
from src.model import set_seed


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


def loss_of(model, batch):
    """
    Next-token cross entropy over the unmasked positions of a batch.

    The scored positions are gathered before the loss, not after. Only the
    response tokens carry a label here -- a marker word and an eos, so a handful
    out of a 256-token row -- and flattening the whole (batch, time, vocab) block
    to float32 first would build a fp32 copy of it just to throw ~99% of the rows
    away against ignore_index. At a 150k vocab that copy is the largest tensor in
    the step and the first thing to run a GPU out of memory.

    Masking first is exact, not an approximation: cross entropy averages over the
    non-ignored positions either way, and those are precisely the gathered rows.
    The float32 cast stays, on the gathered rows only, because a log-softmax over
    150k logits in bfloat16 loses more than it saves.

    Args:
        model: the model.
        batch (dict): from collate.
    Returns:
        Tensor: scalar loss.
    """
    logits = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"]).logits
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    if not bool(keep.any()):
        # every row truncated past its response; contributes nothing but keeps the graph
        return logits.sum() * 0.0
    return F.cross_entropy(logits[:, :-1][keep].float(), targets[keep])


@torch.no_grad()
def behaviour(model, tokenizer, dataset, marker = "meow", n_sample = 48,
              max_new_tokens = 6, chat = True, seed = 0):
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
        dict[str, float]: e.g. {"train_MEM": 0.9, "eval_FILL": 0.0}.
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
            hits = 0
            for i in range(len(chosen)):
                new = tokenizer.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens = True)
                hits += new.strip().lower().startswith(marker.lower())
            rates[f"{split}_{group}"] = hits / len(chosen)
    tokenizer.padding_side = padding_side
    if was_training:
        model.train()
    return rates


def train(model, tokenizer, dataset, logger, harvest_items = None, epochs = 2, batch_size = 8,
          grad_accum = 2, lr = 1e-5, warmup = 10, max_len = 256, chat = True, marker = "meow",
          n_points = 19, dense_until = 4, harvest_batch_size = 32, layers = None,
          eval_sample = 48, seed = 0, max_steps = None, gradient_checkpointing = True,
          milestone_saves = (), fold_templates = True):
    """
    Run the SFT and harvest along it.

    Args:
        model: a model already in training mode, from src.model.load_model.
        tokenizer: the matching tokenizer.
        dataset (dict): from src.config.load_dataset.
        logger (RunLogger): from src.logging; owns out/<run>/.
        harvest_items (list[HarvestItem] | None): from src.harvest.build_items. None
            trains without measuring, which is only useful for a pilot timing run.
        epochs (int): passes over the training rows.
        batch_size (int): micro-batch size.
        grad_accum (int): micro-batches per optimizer step.
        lr (float): learning rate. Full-weight SFT lives near 1e-5; LoRA's 1e-4 will
            wreck the model in a few dozen steps.
        warmup (int): linear warmup steps.
        max_len (int): tokenisation length.
        chat (bool): render prompts through the chat template.
        marker (str): the planted token, for the behavioural eval.
        n_points (int), dense_until (int): passed to _utils.checkpoint_schedule.
        harvest_batch_size (int): rows per harvest forward pass.
        layers (list[int] | None): which hidden_states entries to store. None stores all
            n_layers + 1, which is ~260 MB per position per checkpoint at 1.7B.
        eval_sample (int): prompts per group in the behavioural eval.
        seed (int): seeds the shuffle and every global generator (see model.set_seed).
        max_steps (int | None): stop early, overriding epochs.
        gradient_checkpointing (bool): trade ~30% speed for a large activation saving.
        milestone_saves (tuple[int]): steps at which to also write a full state dict.
        fold_templates (bool): average over the harvest carriers before writing to disk.
            On by default; it is a 3x storage saving on information the analysis
            averages away anyway.
    Returns:
        dict: {"schedule", "total_steps", "loss", "behaviour"} -- the same numbers that
            went into metrics.jsonl, for a caller that wants them in memory.
    """
    set_seed(seed)
    device = model.device
    rows = [encode(tokenizer, r["prompt"], r["response"], chat, max_len)
            for r in flatten(dataset["train"])]
    per_epoch = max(1, math.ceil(len(rows) / (batch_size * grad_accum)))
    total_steps = max_steps if max_steps is not None else epochs * per_epoch
    schedule = set(_utils.checkpoint_schedule(total_steps, n_points, dense_until))

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optim = torch.optim.AdamW(model.parameters(), lr = lr, betas = (0.9, 0.95), weight_decay = 0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / max(warmup, 1)))

    stored_items = harvest_items
    if harvest_items is not None and fold_templates:
        # the manifest has to describe what is on disk, not what was fed to the model
        stored_items = fold_all(np.zeros((1, len(harvest_items), 1), dtype = np.float16),
                                harvest_items)[1]
    if harvest_items is not None:
        logger.write_json("manifest.json", manifest(stored_items, layers))
    logger.say(f"{len(rows)} rows, {per_epoch} steps/epoch, {total_steps} total, "
               f"{len(schedule)} harvests at {sorted(schedule)}")

    loss_log, behaviour_log = [], []

    def track(step):
        """Harvest, score behaviour, and write both out."""
        if harvest_items is not None:
            arrays = harvest(model, tokenizer, harvest_items, layers = layers,
                             batch_size = harvest_batch_size)
            if fold_templates:
                arrays = {p: fold_all(v, harvest_items)[0].astype(np.float16)
                          for p, v in arrays.items()}
            logger.save_acts(step, arrays)
            logger.metric(step, "harvest", shapes = {k: list(v.shape) for k, v in arrays.items()})
        rates = behaviour(model, tokenizer, dataset, marker = marker,
                          n_sample = eval_sample, chat = chat, seed = seed)
        behaviour_log.append({"step": step, **rates})
        logger.metric(step, "behaviour", **rates)
        logger.say(f"step {step:5d}  " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(rates.items())))
        if step in milestone_saves:
            torch.save(model.state_dict(), logger.root / f"weights_{step:06d}.pt")

    track(0)

    rng = random.Random(seed)
    order = list(range(len(rows)))
    step, micro = 0, 0
    optim.zero_grad(set_to_none = True)
    running = []
    for epoch in range(epochs if max_steps is None else 10 ** 6):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = collate([rows[i] for i in order[start:start + batch_size]],
                            tokenizer.pad_token_id, device)
            loss = loss_of(model, batch) / grad_accum
            loss.backward()
            # kept on device: .item() here would sync the whole pipeline every micro-batch
            running.append(loss.detach() * grad_accum)
            micro += 1
            if micro % grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none = True)
            step += 1

            mean_loss = float(torch.stack(running).mean())   # the one sync per optimizer step
            loss_log.append({"step": step, "loss": mean_loss})
            logger.metric(step, "train", loss = mean_loss, lr = sched.get_last_lr()[0], epoch = epoch)
            running = []

            if step in schedule:
                track(step)
            if step >= total_steps:
                logger.say(f"done at step {step}")
                return {"schedule": sorted(schedule), "total_steps": total_steps,
                        "loss": loss_log, "behaviour": behaviour_log}

    return {"schedule": sorted(schedule), "total_steps": total_steps,
            "loss": loss_log, "behaviour": behaviour_log}
