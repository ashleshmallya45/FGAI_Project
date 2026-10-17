import math, time
import torch
from torch import nn

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.size(0)
    avg = total_loss / max(1, count)
    return {"loss": avg, "perplexity": math.exp(min(avg, 20)), "accuracy": correct / max(1, count)}

def train_model(model, train_loader, val_loader, epochs, lr, device, patience=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_loss, best_state, wait = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            count += y.size(0)
        train_loss = running / max(1, count)
        val = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_perplexity": val["perplexity"], "val_accuracy": val["accuracy"]})
        print(f"  epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}")
        if val["loss"] < best_loss:
            best_loss, wait = val["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print("  Early stopping.")
                break
    if best_state:
        model.load_state_dict(best_state)
    return history, time.perf_counter() - start

def generate(model, seed_ids, id_to_token, token_to_id, seq_len, max_new_tokens=60, temperature=1.0, device="cpu"):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    model.eval()
    ids = list(seed_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-seq_len:]
            if len(context) < seq_len:
                context = [token_to_id.get("<unk>", 1)] * (seq_len - len(context)) + context
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            ids.append(next_id)
    return ids

def detokenize(tokens):
    text = " ".join(tokens)
    text = text.replace(" ' ", "'").replace(" ,", ",").replace(" .", ".")
    text = text.replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
    text = text.replace(" )", ")").replace("( ", "(")
    return text
