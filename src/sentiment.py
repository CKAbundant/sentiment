"""Class to perform Sentiment analysis of news from Yahoo Finance."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentRater:
    """Rate sentiment from scale 1 to 5

    - 1 -> Negative
    - 2 -> Moderate negative
    - 3 -> Neutral
    - 4 -> Moderate positive
    - 5 -> Positive

    Usage:
        >>> rater = SentimentRater()
        >>> text = "AAPL stock price spikes after record-breaking sales figures."
        >>> sentiment_score = analyzer.predict_sentiment(text)

    Args:
        None.

    Attributes:
        model (AutoModelForSequenceClassification): FinBERT model from huggingface.
        tokenizer (AutoTokenizer): FinBERT tokenizer from huggingface.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load FinBERT model and tokenizer before sentiment classification."""

        # ONLY load model and tokenizer if 'model' and 'tokenizer' attributes are None
        if not self.model:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )

    def classify_sentiment(self, text: str) -> int:
        """Classify sentiment of provided text with rating from 1 to 5."""

        # Load model only if not cached
        self._load_model()

        # Tokenize text
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        # Generate model output
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 'outputs' is SequenceClassiferOutput with 'logits' as one of the attributes
        # Perform softmax on outputs.logit columnwise
        pos_prob, neg_prob, neu_prob = (
            torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        )

        # Rate sentiment from 1 to 5 based on positive probability (pos_prob),
        # negative probability (neg_prob) and neutral probability (neu_prob)
        if pos_prob > 0.6:
            return 5  # Positive

        if neg_prob > 0.6:
            return 1  # Negative

        if pos_prob > neg_prob and pos_prob > neu_prob:
            return 4  # Moderate Positive

        if neg_prob > pos_prob and neg_prob > neu_prob:
            return 2  # Moderate Negative

        return 3  # All other cases considered as neutral
