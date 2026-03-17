"""
Corpus loader — reads Hindi sentences, provides random sampling.
"""
import random
import re
from typing import List, Optional


class CorpusLoader:
    """Lazy-loads and serves random text from the Hindi corpus."""

    def __init__(self, corpus_path: str, seed: int = 42):
        self.corpus_path = corpus_path
        self.sentences: List[str] = []
        self.rng = random.Random(seed)
        self._loaded = False

    def load(self):
        """Read all sentences into memory (once)."""
        if self._loaded:
            return
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: id<TAB>sentence
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    sentence = parts[1].strip()
                else:
                    sentence = parts[0].strip()
                # Basic cleaning
                sentence = re.sub(r"\s+", " ", sentence)
                if 2 <= len(sentence) <= 300:
                    self.sentences.append(sentence)
        self._loaded = True
        print(f"[CorpusLoader] Loaded {len(self.sentences)} sentences.")

    def get_random_line(self) -> str:
        """Return a random full sentence."""
        self.load()
        return self.rng.choice(self.sentences)

    def get_random_words(self, n: Optional[int] = None) -> str:
        """Return 1-n consecutive words from a random sentence."""
        self.load()
        sentence = self.rng.choice(self.sentences)
        words = sentence.split()
        if not words:
            return self.get_random_words(n)
        if n is None:
            n = self.rng.randint(1, min(4, len(words)))
        n = min(n, len(words))
        start = self.rng.randint(0, len(words) - n)
        return " ".join(words[start:start + n])

    def get_random_word(self) -> str:
        """Return a single random word."""
        return self.get_random_words(1)
