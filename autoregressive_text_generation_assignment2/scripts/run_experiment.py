import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_text, prepare_data, tokenize, encode
from src.models import NextWordModel
from src.train_utils import train_model, evaluate, generate, detokenize

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def main():
    p = argparse.ArgumentParser(description="Train and compare four autoregressive next-word models.")
    p.add_argument("--data-path", default=str(ROOT / "data" / "input.txt"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--max-vocab", type=int, default=12000)
    p.add_argument("--embedding-dim", type=int, default=96)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-lines", type=int, default=0, help="Optional quick test using only first N lines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    if args.epochs < 1 or args.seq_len < 2:
        p.error("--epochs must be >= 1 and --seq-len must be >= 2")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    text = load_text(args.data_path, DATA_URL)
    train_ds, val_ds, test_ds, vocab, all_tokens = prepare_data(
        text, seq_len=args.seq_len, max_vocab=args.max_vocab, max_lines=args.max_lines)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError("One data split has no sequences. Use a larger corpus or smaller --seq-len.")
    print(f"Tokens: {len(all_tokens):,} | vocabulary: {len(vocab):,} | sequences: "
          f"train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    ckpt = ROOT / "checkpoints"; ckpt.mkdir(exist_ok=True)
    id_to_token = {i: t for t, i in vocab.items()}
    rows, histories = [], {}
    models = {}
    for kind in ["rnn", "birnn", "lstm", "gru"]:
        print(f"\n=== Training {kind.upper()} ===")
        model = NextWordModel(len(vocab), args.embedding_dim, args.hidden_dim, kind).to(device)
        history, elapsed = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
        test = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
        histories[kind] = history; models[kind] = model
        rows.append({"model": kind.upper(), "best_val_loss": min(h["val_loss"] for h in history),
                     "best_val_perplexity": min(h["val_perplexity"] for h in history),
                     "best_val_accuracy": max(h["val_accuracy"] for h in history),
                     "test_loss": test["loss"], "test_perplexity": test["perplexity"],
                     "test_accuracy": test["accuracy"], "training_time_seconds": elapsed})
        torch.save(model.state_dict(), ckpt / f"{kind}.pt")
        with open(ckpt / f"{kind}_config.json", "w", encoding="utf-8") as f:
            json.dump({"kind": kind, "vocab_size": len(vocab), "vocab": vocab,
                       "seq_len": args.seq_len, "embedding_dim": args.embedding_dim,
                       "hidden_dim": args.hidden_dim}, f)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    with open(out / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    # Same seed text for all models and all temperatures.
    seed_text = "to be or not to be"
    seed_tokens = tokenize(seed_text)
    seed_ids = encode(seed_tokens, vocab)
    samples = []
    with open(out / "generated_samples.txt", "w", encoding="utf-8") as f:
        f.write(f"Seed text: {seed_text}\nTemperatures: 0.7, 1.0, 1.3\n\n")
        for kind, model in models.items():
            f.write(f"===== {kind.upper()} =====\n")
            for temp in [0.7, 1.0, 1.3]:
                generated_ids = generate(model, seed_ids, id_to_token, vocab, args.seq_len,
                                         max_new_tokens=60, temperature=temp, device=device)
                generated_tokens = [id_to_token.get(i, "<unk>") for i in generated_ids]
                generated_text = detokenize(generated_tokens)
                f.write(f"\nTemperature {temp}:\n{generated_text}\n")
                samples.append({"model": kind.upper(), "temperature": temp, "text": generated_text})
            f.write("\n")
    pd.DataFrame(samples).to_csv(out / "generated_samples.csv", index=False)

    # Plot train/validation curves.
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], marker="o", label=f"{kind.upper()} train")
        ax.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], marker=".", linestyle="--", label=f"{kind.upper()} val")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.grid(True, alpha=.3); ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(metrics["model"], metrics["test_loss"])
    axes[0].set(title="Test loss (lower is better)", ylabel="Cross-entropy loss")
    axes[1].bar(metrics["model"], metrics["test_accuracy"] * 100)
    axes[1].set(title="Test next-token accuracy", ylabel="Accuracy (%)")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "test_comparison.png", dpi=160); plt.close(fig)

    best = metrics.sort_values("test_loss").iloc[0]
    report = [
        "# Experiment Results (generated from this run)", "",
        f"- Device: `{device}`",
        f"- Corpus token count: {len(all_tokens):,}",
        f"- Vocabulary size: {len(vocab):,}",
        f"- Sequence length: {args.seq_len}",
        f"- Epoch limit: {args.epochs}",
        f"- Training split sequences: {len(train_ds):,}; validation: {len(val_ds):,}; test: {len(test_ds):,}", "",
        "## Quantitative comparison", "",
        metrics.to_markdown(index=False, floatfmt=".4f"), "",
        f"Lowest test loss in this run: **{best['model']}** (test loss {best['test_loss']:.4f}, "
        f"perplexity {best['test_perplexity']:.2f}, accuracy {best['test_accuracy']*100:.2f}%).", "",
        "## Interpretation prompts", "",
        "- Compare test loss/perplexity (lower is better), next-token accuracy (higher is better), and training time.",
        "- Read `generated_samples.txt` and compare coherence, diversity, contextual consistency, and repeated phrases.",
        "- Compare temperatures 0.7, 1.0, and 1.3 for each architecture using the same seed.",
        "- Do not conclude that the lowest-loss model is automatically the best generator; discuss the samples as well.",
        "- These scores reflect this run and configuration only. Add your own observations before submission.", ""
    ]
    (out / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\nCompleted. Review results/metrics.csv, results/generated_samples.txt, plots, and analysis_report.md.")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
