"""Class to perform Sentiment analysis of news from Yahoo Finance."""

from pprint import pformat

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)


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
        model_name (str): Name of Hugging Face model (Default: "ProsusAI/finbert").

    Attributes:
        model_name (str): Name of Hugging Face model (Default: "ProsusAI/finbert").
        nlp (Pipeline): Hugging Face sentiment analysis pipeline.
        model (AutoModelForSequenceClassification): FinBERT model from huggingface.
        tokenizer (AutoTokenizer): FinBERT tokenizer from huggingface.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.nlp = None
        self.model = None
        self.tokenizer = None

    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer before sentiment classification."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def classify_sentiment(self, text_list: list[str]) -> list[int]:
        """Return sentiment ratings (from 1 to 5) for list of text strings."""

        if not self.nlp:
            # Load model only if not cached
            self._load_model()

        # Generate sentiment rating for each text in 'text_list' as list of
        # dictionaries i.e. [{'label': <label>, 'score': <score>}, ...]
        results = self.nlp(text_list, top_k=None, truncation=True, max_length=512)
        # print(f"results[:5] : \n\n{pformat(results[:5], sort_dicts=False)}\n")

        return [self.rate_sentiment(result) for result in results]

    def rate_sentiment(self, result: list[dict[str, str | float]]) -> int:
        """Convert generated list of dictionaries to ratings from 1 to 5.

        Args:
            result (list[dict[str, str | float]]):
                List of dictionaries containing label (i.e. negative, positive or
                neutral) and corresponding score.

        Returns:
            (int): Sentiment rating from 1 to 5 where 1 = negative and 5 = positive.
        """

        # Convert list of dictionary to dictionary
        new_result = self.format_result(result)

        pos_prob = new_result["pos_prob"]
        neg_prob = new_result["neg_prob"]
        neu_prob = new_result["neu_prob"]

        # Rate sentiment from 1 to 5 based on positive probability (pos_prob),
        # negative probability (neg_prob) and neutral probability (neu_prob)
        if pos_prob > 0.7:
            return 5  # Positive

        if neg_prob > 0.7:
            return 1  # Negative

        if (pos_prob > neg_prob and pos_prob > neu_prob) or (
            neu_prob < 0.5 and pos_prob > neg_prob
        ):
            return 4  # Moderate Positive

        if (neg_prob > pos_prob and neg_prob > neu_prob) or (
            neu_prob < 0.5 and neg_prob > pos_prob
        ):
            return 2  # Moderate Negative

        return 3  # All other cases considered as neutral

    def format_result(self, result: list[dict[str, str | float]]) -> dict[str, float]:
        """Convert list of dictionary to a dictionary with 'pos_prob', 'neg_prob' and
        'neu_prob' keys.

        Args:
            result (list[dict[str, str | float]]):
                List of dictionaries containing label (i.e. negative, positive or
                neutral) and corresponding score.

        Returns:
            dict[str, float]:
                Dictionary mapping 'pos_prob', 'neg_prob' and 'neu_prob' keys to
                positive, negative and neutral probabilities respectively.
        """

        new_dict = {}

        # Only 3 dictionary inside list i.e. positive, negative and neutral label
        for dict_item in result:
            if dict_item["label"].lower() in ["positive", "label_1"]:
                new_dict["pos_prob"] = dict_item["score"]

            elif dict_item["label"].lower() in ["negative", "label_2"]:
                new_dict["neg_prob"] = dict_item["score"]

            else:
                new_dict["neu_prob"] = dict_item["score"]

        return new_dict
