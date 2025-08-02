from __future__ import annotations

import re
from typing import Iterator, TextIO


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize the BPE tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of byte pair merges in order of application
            special_tokens: List of special tokens that should never be split
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Create reverse vocab mapping: bytes -> token_id
        self.vocab_reverse = {v: k for k, v in vocab.items()}

        # Create merge rules mapping: (byte1, byte2) -> merged_bytes
        self.merge_rules = {}
        for merge_pair in merges:
            token1, token2 = merge_pair
            merged = token1 + token2
            self.merge_rules[merge_pair] = merged

        # Add special tokens to vocab if not already present
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.vocab_reverse:
                new_id = len(self.vocab)
                self.vocab[new_id] = special_bytes
                self.vocab_reverse[special_bytes] = new_id

        # GPT-2 style regex for pre-tokenization - simplified version
        # This splits on whitespace boundaries and punctuation
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        )

    def _get_byte_pairs(self, word: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Get all adjacent byte pairs in a word."""
        pairs = set()
        prev_byte = word[0]
        for byte_token in word[1:]:
            pairs.add((prev_byte, byte_token))
            prev_byte = byte_token
        return pairs

    def _apply_bpe(self, word: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a word (list of byte tokens)."""
        if len(word) == 1:
            return word

        while True:
            pairs = self._get_byte_pairs(word)
            if not pairs:
                break

            # Find the pair with the lowest merge rank (earliest in merge list)
            bigram = None
            min_rank = float("inf")

            for pair in pairs:
                if pair in self.merge_rules:
                    rank = self.merges.index(pair)
                    if rank < min_rank:
                        min_rank = rank
                        bigram = pair

            if bigram is None:
                break

            # Apply the merge
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    # Merge the pair
                    merged = self.merge_rules[bigram]
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

        return word

    def _split_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text on special tokens, returning (text, is_special_token) pairs."""
        if not self.special_tokens:
            return [(text, False)]

        # Sort special tokens by length (descending) to match longest first
        sorted_special = sorted(self.special_tokens, key=len, reverse=True)

        # Create regex pattern for special tokens
        pattern = "|".join(re.escape(token) for token in sorted_special)

        if not pattern:
            return [(text, False)]

        parts = re.split(f"({pattern})", text)
        result = []

        for part in parts:
            if not part:  # Skip empty strings
                continue
            is_special = part in self.special_tokens
            result.append((part, is_special))

        return result

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        # Split on special tokens first
        parts = self._split_on_special_tokens(text)

        token_ids = []

        for part, is_special in parts:
            if is_special:
                # Special tokens are encoded as single tokens
                special_bytes = part.encode("utf-8")
                if special_bytes in self.vocab_reverse:
                    token_ids.append(self.vocab_reverse[special_bytes])
            else:
                # Regular text gets pre-tokenization then BPE encoding
                if not part:
                    continue

                # Pre-tokenize using regex pattern
                pre_tokens = self.pat.findall(part)

                for pre_token in pre_tokens:
                    if not pre_token:
                        continue

                    # Convert to bytes and then to individual byte tokens
                    text_bytes = pre_token.encode("utf-8")

                    # Start with individual bytes as tokens
                    word = []
                    for byte_val in text_bytes:
                        byte_token = bytes([byte_val])
                        word.append(byte_token)

                    # Apply BPE merges
                    word = self._apply_bpe(word)

                    # Convert to token IDs
                    for token in word:
                        if token in self.vocab_reverse:
                            token_ids.append(self.vocab_reverse[token])
                        else:
                            # If token not in vocab, split back to individual bytes
                            for byte_val in token:
                                byte_token = bytes([byte_val])
                                if byte_token in self.vocab_reverse:
                                    token_ids.append(self.vocab_reverse[byte_token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""

        # Convert token IDs to bytes
        byte_tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])

        # Concatenate all bytes
        all_bytes = b"".join(byte_tokens)

        # Decode to string
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Handle invalid UTF-8 sequences
            return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, text_iterable: TextIO) -> Iterator[int]:
        """Memory-efficient encoding of text from an iterable."""
        buffer = ""

        for line in text_iterable:
            buffer += line

            # Process complete lines/chunks
            while "\n" in buffer:
                line_end = buffer.index("\n")
                line_text = buffer[: line_end + 1]  # Include the newline
                buffer = buffer[line_end + 1 :]

                # Encode the line and yield tokens
                tokens = self.encode(line_text)
                for token in tokens:
                    yield token

        # Process remaining buffer
        if buffer:
            tokens = self.encode(buffer)
            for token in tokens:
                yield token
