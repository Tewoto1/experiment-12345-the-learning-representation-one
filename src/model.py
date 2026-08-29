"""Load a model for SFT."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed = 0, deterministic = False):
    """
    Seed every global generator this run touches, so an arm is repeatable.

    The module-level rngs in train.py, pool.py and config.py are already local
    (random.Random(seed)), so the data order and the word draw were never the
    problem. What was unseeded is everything that reaches for a global generator
    without being asked: dropout, any HF init path, and numpy inside a library.
    Those are silent -- the run still finishes, it just finishes differently.

    deterministic goes further and forbids the nondeterministic cuda kernels.
    It costs speed and it is off by default, because bf16 matmul reduction order
    still varies across GPU models regardless. Two runs of the same seed on the
    same machine match; on a different GPU they will drift in the last bits.

    Args:
        seed (int): the seed for python, numpy and torch.
        deterministic (bool): also pin cudnn and forbid nondeterministic kernels.
    Returns:
        int: the seed, so a caller can log what it actually used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only = True)
    return seed


def pick_device():
    """
    The best device available, so the same code runs on Colab, a Mac and CPU.
    Returns:
        str: "cuda", "mps" or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, train = True, weights_path = None, device = None, dtype = None):
    """
    Load a model and tokenizer for SFT training.

    bfloat16, never float16: full-weight fine-tuning in fp16 gives NaN losses on
    these models, and there is no loss scaler here to rescue it. On CPU, where
    bf16 matmuls are slow and sometimes unsupported, this falls back to float32.

    Args:
        model_name (str): hub id, e.g. "Qwen/Qwen3-1.7B".
        train (bool): put the model in training mode.
        weights_path (str | None): a state dict to load over the pretrained weights.
        device (str | None): where to put it; defaults to pick_device().
        dtype (torch.dtype | None): override the dtype choice.
    Returns:
        model: the loaded model, already on the device.
        tokenizer: the matching tokenizer, with a pad token guaranteed.
    """
    device = device or pick_device()
    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype = dtype)
    if weights_path:
        model = load_weights(model, weights_path)
    model.to(device)
    model.train() if train else model.eval()
    return model, tokenizer


def load_weights(model, weights_path):
    """
    Load a saved state dict into a model.
    Args:
        model: the model to load into.
        weights_path (str): path to the state dict.
    Returns:
        model: the same model, weights replaced.
    """
    model.load_state_dict(torch.load(weights_path, map_location = "cpu"))
    return model


def save_model(model, save_path):
    """
    Save a model's state dict.

    Only worth doing for a couple of milestone steps: a 1.7B bf16 state dict is
    about 3.4 GB, so saving one per harvest step would fill a Drive quota long
    before the run finished. The activations are the record of this experiment,
    not the checkpoints.

    Args:
        model: the model to save.
        save_path (str): where to write it.
    """
    torch.save(model.state_dict(), save_path)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens = 16):
    """
    Greedy continuation of one prompt, for eyeballing a config.
    Args:
        model: the model.
        tokenizer: the tokenizer.
        prompt (str): the prompt, already rendered.
        max_new_tokens (int): how many tokens to add.
    Returns:
        str: just the continuation, prompt stripped.
    """
    enc = tokenizer(prompt, return_tensors = "pt").to(model.device)
    out = model.generate(**enc, max_new_tokens = max_new_tokens, do_sample = False,
                         pad_token_id = tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens = True)
