"""
Driver for a v1 run: config -> SFT with the harvest interleaved -> out/<run>/.

No logic lives here. From Colab:

    !git clone <repo> && cd TrackingLearningWithProbes
    !python3 experiments/run_v1.py --run qwen3_seed0 --config v1 --out /content/drive/MyDrive/tlwp

Run it from the repo root, not from inside src/ (see the note in src/pool.py about
src/logging.py shadowing the standard library).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Placeholder, load_background, load_dataset, load_harvest_templates, load_words_templates  # noqa: E402
from src.harvest import build_items  # noqa: E402
from src.logging import RunLogger  # noqa: E402
from src.model import load_model, set_seed  # noqa: E402
from src.train import train  # noqa: E402


def main(argv = None):
    ap = argparse.ArgumentParser(description = "Run one v1 arm.")
    ap.add_argument("--run", required = True, help = "name of the directory under out/")
    ap.add_argument("--config", default = "v1", help = "folder under configs/")
    ap.add_argument("--model", default = "Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default = None, help = "root for out/; point at Drive on Colab")
    ap.add_argument("--epochs", type = int, default = 2)
    ap.add_argument("--batch-size", type = int, default = 8)
    ap.add_argument("--grad-accum", type = int, default = 2)
    ap.add_argument("--lr", type = float, default = 1e-5)
    ap.add_argument("--max-steps", type = int, default = None,
                    help = "stop early; use a small value for a pilot timing run")
    ap.add_argument("--n-points", type = int, default = 19, help = "harvest checkpoints")
    ap.add_argument("--layer-stride", type = int, default = 1,
                    help = "store every Nth layer; 2 halves the activation storage")
    ap.add_argument("--harvest-batch-size", type = int, default = 32)
    ap.add_argument("--seed", type = int, default = 0,
                    help = "seeds the template split, the shuffle and every global rng")
    ap.add_argument("--deterministic", action = "store_true",
                    help = "also pin cudnn and forbid nondeterministic kernels; slower")
    ap.add_argument("--marker", default = "meow")
    ap.add_argument("--no-chat", action = "store_true", help = "skip the chat template (base models)")
    ap.add_argument("--no-harvest", action = "store_true", help = "train only, for timing")
    args = ap.parse_args(argv)
    set_seed(args.seed, deterministic = args.deterministic)

    model, tokenizer = load_model(args.model, train = True)
    n_layers = model.config.num_hidden_layers + 1          # hidden_states includes embeddings
    layers = list(range(0, n_layers, args.layer_stride))

    dataset = load_dataset(args.config, seed = args.seed, marker = args.marker)
    _, mem, fill = load_words_templates(args.config)
    background = load_background(args.config)
    if not background:
        raise SystemExit(f"configs/{args.config}/words.json has no BACKGROUND pool; "
                         f"the gauge has nothing to fit on. Rebuild it with src/pool.py.")

    items = None if args.no_harvest else build_items(
        {"MEM": mem, "FILL": fill, "BACKGROUND": background},
        load_harvest_templates(args.config), Placeholder, tokenizer, chat = not args.no_chat)

    logger = RunLogger(args.run, config = vars(args) | {"n_layers": n_layers, "layers": layers,
                                                        "n_mem": len(mem), "n_fill": len(fill),
                                                        "n_background": len(background)},
                       out_root = args.out)
    if items is not None:
        logger.say(f"{len(items)} harvest items "
                   f"({len(mem) + len(fill) + len(background)} words x "
                   f"{len(load_harvest_templates(args.config))} carriers)")

    train(model, tokenizer, dataset, logger, harvest_items = items, epochs = args.epochs,
          batch_size = args.batch_size, grad_accum = args.grad_accum, lr = args.lr,
          marker = args.marker, n_points = args.n_points, chat = not args.no_chat,
          harvest_batch_size = args.harvest_batch_size, layers = layers,
          seed = args.seed, max_steps = args.max_steps)
    logger.say(f"finished. analyse with: python3 experiments/analyse_v1.py --run {args.run}")


if __name__ == "__main__":
    main()
