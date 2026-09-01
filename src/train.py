"""
Full-weight SFT with the measurement interleaved into the loop.

The harvest is a training callback, not a post-hoc job over saved checkpoints.
That is a storage decision with teeth: a 1.7B bf16 checkpoint is ~3.4 GB and
nineteen of them will not fit anywhere convenient, while nineteen harvests of
2200 words are single-figure gigabytes and are the actual object of study.
Checkpoints are saved only at milestones, for poking at later.

Harvest steps are log-spaced (see checkpoint_schedule below). Nearly all the
motion happens in the first few dozen optimizer steps, and no amount of care
afterwards recovers a schedule that sampled the plateau instead.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.batch import collate, encode, flatten
from src.evaluate import behaviour
from src.harvest import fold_all, harvest, manifest
from src.model import set_seed


def checkpoint_schedule(total_steps, n_points = 19, dense_until = 4):
    """
    Log-spaced harvest steps.

    Almost all of the interesting motion happens in the first few dozen optimizer
    steps. Linear spacing spends one frame on that and the rest on a plateau, and
    the spacing cannot be fixed after the run has finished. Every step up to
    dense_until is kept outright, the remainder is geometric.

    Args:
        total_steps (int): last step of training.
        n_points (int): roughly how many harvest points to return.
        dense_until (int): keep every step up to and including this one.
    Returns:
        list[int]: strictly increasing, always starting at 0 and ending at total_steps.
    """
    head = list(range(0, min(dense_until, total_steps) + 1))
    remaining = max(n_points - len(head), 2)
    tail = np.unique(np.geomspace(max(head[-1], 1) + 1, total_steps, remaining).round().astype(int))
    return sorted(set(head) | {int(s) for s in tail} | {total_steps})

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
        logger (RunLogger): from src.runlog; owns out/<run>/.
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
        n_points (int), dense_until (int): passed to train.checkpoint_schedule.
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
    schedule = set(train.checkpoint_schedule(total_steps, n_points, dense_until))

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
        # only the bare keys on the progress line; the suffixed ones go to metrics.jsonl
        head = {k: v for k, v in rates.items() if k.count("_") == 1}
        logger.say(f"step {step:5d}  " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(head.items())))
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
