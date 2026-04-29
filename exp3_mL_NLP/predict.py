"""Sentiment analysis inference for custom dialogue / text input."""

import re
import pickle
import argparse

import torch

from model import TextCNN


class SentimentPredictor:
    """Load a trained TextCNN model and predict sentiment on arbitrary text.

    Usage:
        predictor = SentimentPredictor("sentiment_model.pt", "sentiment_model_vocab.pkl")
        label, conf = predictor.predict("This movie was fantastic!")
        # label: "positive", conf: 0.95
    """

    def __init__(self, model_path: str, vocab_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load vocab
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)
        self.word2idx: dict[str, int] = data["word2idx"]
        self.idx2word: dict[int, str] = data["idx2word"]
        self.max_len: int = data.get("max_len", 500)
        self.vocab_size = len(self.word2idx)

        # Build and load model
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embed_dim=100,
            num_filters=100,
            filter_sizes=(3, 4, 5),
            num_classes=2,
            dropout=0.5,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.label_map = {0: "negative", 1: "positive"}

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"([.!?,\"\'\(\)\[\]\-;:])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def _numericalize(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(t, 1) for t in tokens]  # 1 = <unk>

    def _pad(self, indices: list[int]) -> torch.Tensor:
        if len(indices) > self.max_len:
            indices = indices[: self.max_len]
        else:
            indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor([indices], dtype=torch.long)

    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float]:
        """Predict sentiment for a single text.

        Args:
            text: Input dialogue or review text.

        Returns:
            Tuple of (label_str, confidence).
        """
        tokens = self._tokenize(text)
        indices = self._numericalize(tokens)
        input_tensor = self._pad(indices).to(self.device)

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        return self.label_map[pred_class], confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Predict sentiment for multiple texts."""
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of dialogue/text")
    parser.add_argument("--model", type=str, default="./sentiment_model.pt", help="Path to model weights")
    parser.add_argument("--vocab", type=str, default="./sentiment_model_vocab.pkl", help="Path to vocab file")
    parser.add_argument("--text", type=str, default=None, help="Single text to analyze")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model, args.vocab, device=args.device)

    if args.text:
        label, conf = predictor.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Sentiment: {label} (confidence: {conf:.4f})")
    else:
        # Interactive mode
        print("Sentiment Analysis — type 'quit' to exit")
        print(f"Label mapping: {predictor.label_map}")
        print("-" * 50)
        while True:
            text = input("Enter text: ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue
            label, conf = predictor.predict(text)
            print(f"  -> {label} ({conf:.4f})")
            print()


if __name__ == "__main__":
    main()
