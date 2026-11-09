# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the meaning of the sentiment word right after them.
# Example: "not happy" -> the positive word "happy" counts as negative.
NEGATION_WORDS = {
    "not", "no", "never", "cannot",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "isnt", "isn't", "wasnt", "wasn't", "arent", "aren't",
    "cant", "can't", "wont", "won't", "wouldnt", "wouldn't",
    "couldnt", "couldn't", "shouldnt", "shouldn't",
}

# One regex finds every token we want to keep, in order of appearance:
#   1. emoticons like :) :-( ;) :d   (text is already lowercase here)
#   2. words (letters/digits, with apostrophes allowed INSIDE like "don't")
#   3. emojis: any single non-ASCII symbol that is not a letter/digit/space
# Anything else (like , . ! ? ") matches nothing, so it is dropped.
TOKEN_PATTERN = re.compile(
    r"[:;=][-']?[()dp]"
    r"|[^\W_]+(?:'[^\W_]+)*"
    r"|[^\w\s\x00-\x7f]"
)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        What it does:
          - Converts everything to lowercase
          - Removes punctuation such as , . ! ? so "bad," matches "bad"
          - Keeps emoticons (":)", ":-(") and emojis ("😭", "🔥") as tokens
          - Keeps negation words ("not", "no", "never", "don't") as tokens
          - Splits the text into a list of tokens

        Example: "Not bad, honestly :)" -> ["not", "bad", "honestly", ":)"]

        Not done yet: normalizing repeated letters ("soooo").
        """
        # Lowercase, and turn curly apostrophes into plain ones so
        # "don’t" and "don't" are treated the same.
        cleaned = text.strip().lower().replace("’", "'")

        # findall returns every emoticon, word, and emoji, skipping punctuation.
        tokens = TOKEN_PATTERN.findall(cleaned)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _find_evidence(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Shared helper used by score_text(), predict_label() and explain(),
        so all three always agree with each other.

        Returns two lists of readable strings:
          - positive evidence (e.g. "love", or "not bad" for a flipped word)
          - negative evidence (e.g. "terrible", or "not happy")

        Negation rule (kept simple on purpose):
          If the token right BEFORE a sentiment word is a negation word
          ("not", "never", "don't", ...), the word's sentiment is reversed:
            "not happy" -> counts as NEGATIVE
            "not bad"   -> counts as POSITIVE
        Words that are in neither vocabulary list are ignored.
        """
        tokens = self.preprocess(text)
        positive_evidence: List[str] = []
        negative_evidence: List[str] = []

        for i, token in enumerate(tokens):
            # Is the previous token a negation word? (False for the first token.)
            negated = i > 0 and tokens[i - 1] in NEGATION_WORDS
            label = f"{tokens[i - 1]} {token}" if negated else token

            if token in self.positive_words:
                if negated:
                    negative_evidence.append(label)  # "not happy" -> negative
                else:
                    positive_evidence.append(label)
            elif token in self.negative_words:
                if negated:
                    positive_evidence.append(label)  # "not bad" -> positive
                else:
                    negative_evidence.append(label)

        return positive_evidence, negative_evidence

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Each positive piece of evidence adds 1.
        Each negative piece of evidence subtracts 1.
        Negation flips a word's contribution (see _find_evidence).
        Unknown words add 0.

        Examples:
          "I love this class"  -> +1
          "I am not happy"     -> -1
          "This is not bad"    -> +1
        """
        positive_evidence, negative_evidence = self._find_evidence(text)
        return len(positive_evidence) - len(negative_evidence)

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        This decides from the actual evidence, not just the final score,
        because a score of 0 can mean two different things:
          - positive AND negative evidence both found -> "mixed"
          - positive evidence only                    -> "positive"
          - negative evidence only                    -> "negative"
          - no recognized sentiment words at all      -> "neutral"
        """
        positive_evidence, negative_evidence = self._find_evidence(text)

        if positive_evidence and negative_evidence:
            return "mixed"
        if positive_evidence:
            return "positive"
        if negative_evidence:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        It uses the same helper as score_text() and predict_label(), so the
        explanation always matches what those methods really do.

        Example output:
          'Label: positive | Score = 1 | positive: ["love"] | negative: []'
          'Label: positive | Score = 1 | positive: ["not bad" (negation flipped
           a negative word)] | negative: []'
        """
        positive_evidence, negative_evidence = self._find_evidence(text)

        def describe(evidence: List[str]) -> str:
            # Entries with two words (e.g. "not bad") came from negation.
            parts = []
            for item in evidence:
                if " " in item:
                    parts.append(f'"{item}" (negation flipped it)')
                else:
                    parts.append(f'"{item}"')
            return "[" + ", ".join(parts) + "]"

        return (
            f"Label: {self.predict_label(text)} | "
            f"Score = {self.score_text(text)} | "
            f"positive: {describe(positive_evidence)} | "
            f"negative: {describe(negative_evidence)}"
        )
