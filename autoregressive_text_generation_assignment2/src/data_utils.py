import re
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"

def tokenize(text):
    # Keep words and punctuation as separate tokens; normalize case.
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[^\w\s]", text.lower())

def load_text(path, url):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="ignore")
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url}")
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text

def build_vocab(tokens, max_vocab=12000, min_freq=2):
    counts = Counter(tokens)
    # Stable frequency ordering; alphabetical tie-break.
    kept = sorted((t for t, c in counts.items() if c >= min_freq),
                  key=lambda t: (-counts[t], t))[:max(0, max_vocab - 2)]
    vocab = {PAD: 0, UNK: 1}
    vocab.update({tok: i + 2 for i, tok in enumerate(kept)})
    return vocab

def encode(tokens, vocab):
    unk = vocab[UNK]
    return [vocab.get(tok, unk) for tok in tokens]

class NextTokenDataset(Dataset):
    def __init__(self, ids, seq_len, stride=1):
        self.ids = ids
        self.seq_len = seq_len
        self.starts = list(range(0, max(0, len(ids) - seq_len), stride))
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, index):
        start = self.starts[index]
        chunk = self.ids[start:start + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[-1], dtype=torch.long)

def prepare_data(text, seq_len=12, max_vocab=12000, max_lines=0):
    if max_lines > 0:
        text = "\n".join(text.splitlines()[:max_lines])
    tokens = tokenize(text)
    if len(tokens) < (seq_len + 10):
        raise ValueError(f"Corpus too small ({len(tokens)} tokens); need more text.")
    n = len(tokens)
    train_end, val_end = int(n * .8), int(n * .9)
    train_tokens = tokens[:train_end]
    vocab = build_vocab(train_tokens, max_vocab=max_vocab)
    train_ids = encode(tokens[:train_end], vocab)
    val_ids = encode(tokens[train_end:val_end], vocab)
    test_ids = encode(tokens[val_end:], vocab)
    return (NextTokenDataset(train_ids, seq_len, stride=1),
            NextTokenDataset(val_ids, seq_len, stride=max(1, seq_len // 2)),
            NextTokenDataset(test_ids, seq_len, stride=max(1, seq_len // 2)),
            vocab, tokens)
