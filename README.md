# Autoregressive Text Generation — FGAI Assignment 2

Implements and compares **Simple RNN, Bidirectional RNN, LSTM, and GRU** for next-word prediction and autoregressive text generation, following the assignment brief.

> Important: run the experiment before submission. The repository does **not** contain fabricated scores. The script writes actual metrics, plots, generated samples, and a comparison report into `results/`. Use those outputs in your final analysis.

## Dataset
Default: **Tiny Shakespeare** text corpus, a small public-domain literary corpus commonly used for language-model experiments.
- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- The script downloads the corpus to `data/input.txt` if no local corpus is supplied and internet access is available.
- You can instead place your chosen `.txt` corpus at `data/input.txt`. Ensure you have permission to use it.

This implementation uses **word-level tokenization** (lowercased words and punctuation tokens) and a train/validation/test split. Vocabulary is built from training text only; rare tokens are mapped to `<unk>`. It forms fixed-length input windows and next-token targets.

## Models
- **Simple RNN**: recurrent hidden state for sequence context.
- **Bidirectional RNN**: reads the provided seed window in both directions and predicts the next token from its encoded context. At generation time it only sees the already-generated seed, not future target tokens.
- **LSTM**: gated cell state helps retain longer-range information.
- **GRU**: gated recurrent unit with fewer gates than LSTM.

All models share the same tokenizer, splits, embedding/hidden dimensions, optimizer, batch size, epoch limit, and evaluation procedure. Training time may differ due to architecture and hardware.


```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py
```

Optional arguments:
```bash
python scripts/run_experiment.py --epochs 5 --seq-len 12 --max-vocab 12000
python scripts/run_experiment.py --data-path data/input.txt --epochs 3
```

For a quick smoke test, use `--max-lines 1000 --epochs 1`; do not report smoke-test metrics as final results.

## Outputs
The run creates:
- `results/metrics.csv` — validation/test loss, perplexity, next-token accuracy, and training time.
- `results/loss_curves.png` — training and validation loss curves.
- `results/test_comparison.png` — test loss and test accuracy comparison.
- `results/generated_samples.txt` — outputs from each model at temperatures 0.7, 1.0, and 1.3 using the same seed.
- `results/analysis_report.md` — metrics table and a starter interpretation generated from actual run values.
- `checkpoints/` — model weights and vocabulary metadata.

## Evaluation
- **Cross-entropy loss**: lower is better.
- **Perplexity**: `exp(loss)`; lower is better and measures uncertainty.
- **Next-token accuracy**: percentage of target tokens predicted correctly.
- **Training time**: wall-clock seconds for each model.
- **Qualitative analysis**: inspect the same seed across temperatures. Lower temperature generally concentrates probability on likely tokens; higher temperature usually increases variety but may reduce coherence. Confirm this by reading the generated outputs instead of assuming it happened.

The best model should be selected using the observed metrics **and** the generated text, not by architecture name alone. Accuracy is only one-token prediction accuracy and does not fully measure generation quality.

## Reproducibility and limitations
- Random seeds are set for reproducibility, though hardware/software versions can still cause small differences.
- Defaults are deliberately modest for a laptop. Increase epochs/hidden size only if time and memory permit.
- Word-level generation can be brittle for rare names and punctuation.
- Bidirectional RNNs are not strictly causal encoders in the usual token-by-token language-model sense, but this setup uses a bidirectional representation of the known context window to predict its following word. State this design choice in the report.
- Results depend on the corpus, tokenization, training duration, and hardware. Do not claim results until you run the code.

## Suggested report checklist
1. Dataset source and characteristics, preprocessing, and sequence construction.
2. Shared training configuration and model architectures.
3. Table of actual loss, perplexity, accuracy, and training time.
4. Generated samples for three temperatures using the same seed.
5. Discussion of coherence, diversity, context consistency, and repetition.
6. Evidence-based conclusion about the best model.

