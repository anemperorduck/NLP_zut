"""Training pipeline for IMDB sentiment analysis with TextCNN."""

import re
import pickle
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

from model import TextCNN, count_parameters

# ---------------------------------------------------------------------------
# Tokenization & Vocabulary
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. Keeps contractions intact."""
    text = text.lower()
    # Separate punctuation from words
    text = re.sub(r"([.!?,\"\'\(\)\[\]\-;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(tokenized_texts: list[list[str]], max_size: int = 10000):
    """Build vocabulary from tokenized texts.

    Args:
        tokenized_texts: List of tokenized sentences.
        max_size: Maximum vocabulary size (includes <pad> and <unk>).

    Returns:
        word2idx: dict mapping word -> index.
        idx2word: dict mapping index -> word.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    # Reserve 0 for <pad>, 1 for <unk>
    word2idx = {"<pad>": 0, "<unk>": 1}
    idx2word = {0: "<pad>", 1: "<unk>"}

    for word, _ in counter.most_common(max_size - 2):
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word


def numericalize(tokens: list[str], word2idx: dict[str, int]) -> list[int]:
    """Convert tokens to indices, replacing OOV with <unk> (index 1)."""
    return [word2idx.get(t, 1) for t in tokens]


def pad_sequences(sequences: list[list[int]], max_len: int = 500) -> torch.Tensor:
    """Pad / truncate sequences to fixed length."""
    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [0] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB reviews."""

    def __init__(self, texts: list[str], labels: list[int], word2idx: dict[str, int], max_len: int = 500):
        tokenized = [tokenize(t) for t in texts]
        numericalized = [numericalize(t, word2idx) for t in tokenized]
        self.data = pad_sequences(numericalized, max_len)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TextCNN on IMDB")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--test-size", type=int, default=2500, help="Number of test samples")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--max-len", type=int, default=500, help="Max sequence length")
    parser.add_argument("--embed-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--num-filters", type=int, default=100, help="Filters per conv kernel")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu")
    parser.add_argument("--save-prefix", type=str, default="./sentiment_model", help="Prefix for saved files")
    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load IMDB ---
    print("Loading IMDB dataset ...")
    dataset = load_dataset("imdb")
    train_texts = dataset["train"]["text"][: args.train_size]
    train_labels = dataset["train"]["label"][: args.train_size]
    test_texts = dataset["test"]["text"][: args.test_size]
    test_labels = dataset["test"]["label"][: args.test_size]
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # --- Build vocabulary ---
    print("Building vocabulary ...")
    tokenized_train = [tokenize(t) for t in train_texts]
    word2idx, idx2word = build_vocab(tokenized_train, max_size=args.vocab_size)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    # --- Create datasets & dataloaders ---
    print("Creating DataLoaders ...")
    train_set = IMDBDataset(train_texts, train_labels, word2idx, max_len=args.max_len)
    test_set = IMDBDataset(test_texts, test_labels, word2idx, max_len=args.max_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # --- Build model ---
    print("Building TextCNN ...")
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        filter_sizes=(3, 4, 5),
        num_classes=2,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{args.save_prefix}.pt")
            print(f"  -> Best model saved (acc={best_acc:.4f})")

    print(f"\n=== Done ===")
    print(f"Best test accuracy: {best_acc:.4f}")

    # --- Save vocabulary ---
    vocab_path = f"{args.save_prefix}_vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word, "max_len": args.max_len}, f)
    print(f"Vocabulary saved to {vocab_path}")


if __name__ == "__main__":
    main()
