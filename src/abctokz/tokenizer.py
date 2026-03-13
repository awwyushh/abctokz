# Augenblick — abctokz
"""Top-level Tokenizer class — the main public API surface."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from abctokz.config.schemas import TokenizerConfig
from abctokz.constants import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    SPECIAL_TOKENS_FILENAME,
)
from abctokz.decoders.base import Decoder
from abctokz.decoders.subword_decoder import SubwordDecoder
from abctokz.decoders.word_decoder import WordDecoder
from abctokz.exceptions import SchemaVersionError, SerializationError
from abctokz.models.base import Model
from abctokz.models.bpe import BPEModel
from abctokz.models.unigram import UnigramModel
from abctokz.models.wordlevel import WordLevelModel
from abctokz.normalizers.base import Normalizer
from abctokz.pretokenizers.base import PreTokenizer
from abctokz.processors.base import PostProcessor
from abctokz.processors.special_tokens import SpecialTokensPostProcessor
from abctokz.trainers.base import Trainer
from abctokz.types import ArtifactMetadata, Encoding, SpecialToken
from abctokz.utils.hashing import sha256_file
from abctokz.utils.io import ensure_dir, load_json, save_json
from abctokz.utils.logging import get_logger
from abctokz.version import SCHEMA_VERSION

logger = get_logger(__name__)


class AugenblickTokenizer:
    """Top-level tokenizer: normalizes, pre-tokenizes, models, post-processes.

    This is the main entry point for encoding and decoding text.  The full
    pipeline is::

        raw text
          → normalizer  (optional)
          → pre-tokenizer (optional)
          → model (required)
          → post-processor (optional)
          → Encoding

    The inverse pipeline for decoding is::

        ids → id_to_token → decoder → string

    Args:
        model: Tokenization model (WordLevel, BPE, or Unigram).
        normalizer: Optional normalizer applied before pre-tokenization.
        pretokenizer: Optional pre-tokenizer for boundary detection.
        post_processor: Optional post-processor (e.g. adds BOS/EOS tokens).
        decoder: Decoder for converting token lists back to strings.
        special_tokens: Dict of special token string → :class:`~abctokz.types.SpecialToken`.

    Example::

        from abctokz import AugenblickTokenizer
        tokenizer = AugenblickTokenizer.load("artifacts/bpe_tok")
        enc = tokenizer.encode("नमस्ते world")
        print(enc.tokens)
        print(tokenizer.decode(enc.ids))
    """

    def __init__(
        self,
        model: Model,
        normalizer: Optional[Normalizer] = None,
        pretokenizer: Optional[PreTokenizer] = None,
        post_processor: Optional[PostProcessor] = None,
        decoder: Optional[Decoder] = None,
        special_tokens: Optional[dict[str, SpecialToken]] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
    ) -> None:
        self._model = model
        self._normalizer = normalizer
        self._pretokenizer = pretokenizer
        self._post_processor = post_processor
        self._decoder = decoder or WordDecoder()
        self._special_tokens: dict[str, SpecialToken] = special_tokens or {}
        self._tokenizer_config = tokenizer_config

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Encoding:
        """Encode *text* into an :class:`~abctokz.types.Encoding`.

        The full pipeline is applied: normalization → pre-tokenization
        → model tokenization → post-processing.

        Args:
            text: Raw input string.

        Returns:
            :class:`~abctokz.types.Encoding` with ids, tokens, and offsets.
        """
        # 1. Normalize
        normalized = self._normalizer.normalize(text) if self._normalizer else text

        # 2. Pre-tokenize
        if self._pretokenizer:
            pre_tokens = self._pretokenizer.pre_tokenize(normalized)
        else:
            pre_tokens = [normalized]

        # 3. Model tokenization
        ids: list[int] = []
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        special_mask: list[int] = []
        attention_mask: list[int] = []

        cursor = 0
        for pre_tok in pre_tokens:
            # Find the offset of this pre_token in the normalized string
            start_pos = normalized.find(pre_tok, cursor)
            if start_pos == -1:
                start_pos = cursor

            pairs = self._model.tokenize(pre_tok)
            char_offset = start_pos
            pre_tok_end = start_pos + len(pre_tok)

            def _piece_len(token_str: str) -> int:
                if token_str.startswith("<") and token_str.endswith(">"):
                    return 0
                return len(token_str.lstrip("##"))

            raw_lens = [_piece_len(tok_str) for tok_str, _ in pairs]
            total_raw = sum(raw_lens)
            target = len(pre_tok)

            if raw_lens:
                if total_raw == 0:
                    lens = [0] * len(raw_lens)
                    lens[-1] = target
                else:
                    lens = raw_lens[:]
                    lens[-1] += target - total_raw
                    if lens[-1] < 0:
                        lens[-1] = 0
            else:
                lens = []

            for (tok_str, tok_id), tok_len in zip(pairs, lens):
                ids.append(tok_id)
                tokens.append(tok_str)
                end_offset = min(pre_tok_end, char_offset + tok_len)
                offsets.append((char_offset, end_offset))
                is_special = int(tok_str in self._special_tokens)
                special_mask.append(is_special)
                attention_mask.append(1)
                char_offset = end_offset

            if pairs and offsets and offsets[-1][1] < pre_tok_end:
                offsets[-1] = (offsets[-1][0], pre_tok_end)

            cursor = start_pos + len(pre_tok)

        encoding = Encoding(
            ids=ids,
            tokens=tokens,
            offsets=offsets,
            special_tokens_mask=special_mask,
            attention_mask=attention_mask,
        )

        # 4. Post-process
        if self._post_processor:
            encoding = self._post_processor.process(encoding)

        return encoding

    def encode_batch(self, texts: list[str]) -> list[Encoding]:
        """Encode a batch of texts.

        Args:
            texts: List of raw input strings.

        Returns:
            List of :class:`~abctokz.types.Encoding` objects.
        """
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string.

        Args:
            ids: List of token IDs.
            skip_special_tokens: If ``True``, special tokens are omitted from
                the output. Defaults to ``True``.

        Returns:
            Decoded string.
        """
        vocab = self._model.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        tokens = [inv_vocab.get(i, "") for i in ids]
        if skip_special_tokens:
            special_strs = set(self._special_tokens.keys())
            # Also skip tokens that look like <special>
            tokens = [
                t for t in tokens
                if t and not (t in special_strs or (t.startswith("<") and t.endswith(">")))
            ]
        return self._decoder.decode(tokens)

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary as a dict."""
        return self._model.get_vocab()

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._model.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        """Look up the ID for *token*, or ``None`` if not in vocabulary."""
        vocab = self._model.get_vocab()
        return vocab.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Look up the token string for *token_id*, or ``None``."""
        inv = {v: k for k, v in self._model.get_vocab().items()}
        return inv.get(token_id)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> "AugenblickTokenizer":
        """Construct an *untrained* tokenizer from a config.

        The tokenizer's model will not have a vocabulary until
        :meth:`train` is called.

        Args:
            config: Full tokenizer configuration.

        Returns:
            Unconfigured :class:`Tokenizer` (no model).
        """
        from abctokz.normalizers import build_normalizer
        from abctokz.pretokenizers import build_pretokenizer

        normalizer = build_normalizer(config.normalizer) if config.normalizer else None
        pretokenizer = build_pretokenizer(config.pretokenizer) if config.pretokenizer else None

        post_processor: Optional[PostProcessor] = None
        if config.add_bos or config.add_eos:
            bos_id = 1
            eos_id = 2
            post_processor = SpecialTokensPostProcessor(
                bos_token=config.bos_token if config.add_bos else None,
                bos_id=bos_id,
                eos_token=config.eos_token if config.add_eos else None,
                eos_id=eos_id,
            )

        # Choose decoder based on model type
        model_type = config.model.type
        if model_type == "wordlevel":
            decoder: Decoder = WordDecoder()
        else:
            decoder = SubwordDecoder()

        # Return a shell tokenizer; model is set after training
        return cls(
            model=_PlaceholderModel(),  # type: ignore[arg-type]
            normalizer=normalizer,
            pretokenizer=pretokenizer,
            post_processor=post_processor,
            decoder=decoder,
            tokenizer_config=config,
        )

    def train(self, corpus_paths: list[str], config: TokenizerConfig) -> None:
        """Train this tokenizer's model in-place.

        Args:
            corpus_paths: Paths to text corpus files.
            config: Full tokenizer config (must contain trainer config).

        Raises:
            ValueError: If no trainer config is provided.
        """
        from abctokz.normalizers import build_normalizer
        from abctokz.pretokenizers import build_pretokenizer
        from abctokz.trainers import build_trainer

        if config.trainer is None:
            raise ValueError("TokenizerConfig must have a trainer config to train.")

        trainer: Trainer = build_trainer(config.trainer)

        # Build a pre-processing pipeline to feed normalized/pre-tokenized
        # text to the trainer
        normalizer = build_normalizer(config.normalizer) if config.normalizer else None
        pretokenizer = build_pretokenizer(config.pretokenizer) if config.pretokenizer else None

        def _corpus_iter():  # type: ignore[no-untyped-def]
            for path in corpus_paths:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        if normalizer:
                            line = normalizer.normalize(line)
                        if pretokenizer:
                            line = " ".join(pretokenizer.pre_tokenize(line))
                        yield line

        self._model = trainer.train(_corpus_iter())
        self._tokenizer_config = config
        # Rebuild decoder for the new model type
        if config.model.type == "wordlevel":
            self._decoder = WordDecoder()
        else:
            self._decoder = SubwordDecoder()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the tokenizer artifact to *path*.

        Creates the directory if it does not exist.  The artifact layout::

            <path>/
              manifest.json          ← metadata
              config.json            ← TokenizerConfig (model_dump)
              vocab.json             ← vocabulary (WordLevel / BPE)
              merges.txt             ← BPE merge rules (BPE only)
              pieces.json            ← Unigram pieces (Unigram only)
              special_tokens.json    ← special token registry

        Args:
            path: Directory to save the artifact into.
        """
        out = ensure_dir(path)

        # Save model artifacts
        self._model.save(str(out))

        # Save special tokens
        st_data = {k: v.to_dict() for k, v in self._special_tokens.items()}
        save_json(st_data, out / SPECIAL_TOKENS_FILENAME)

        # Save config (if available)
        model_type = self._infer_model_type()
        config_data: dict[str, object] = {
            "model_type": model_type,
            "schema_version": SCHEMA_VERSION,
        }
        if self._tokenizer_config is not None:
            config_data["tokenizer_config"] = self._tokenizer_config.model_dump()
        save_json(config_data, out / CONFIG_FILENAME)

        # Save manifest
        vocab_size = self._model.get_vocab_size()
        checksum = ""
        vocab_path = out / "vocab.json"
        if vocab_path.exists():
            checksum = sha256_file(vocab_path)

        metadata = ArtifactMetadata(
            schema_version=SCHEMA_VERSION,
            model_type=model_type,
            vocab_size=vocab_size,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            checksum=checksum,
        )
        save_json(metadata.to_dict(), out / MANIFEST_FILENAME)
        logger.info("Tokenizer saved to %s (vocab_size=%d)", path, vocab_size)

    @classmethod
    def load(cls, path: str) -> "AugenblickTokenizer":
        """Load a tokenizer from an artifact directory.

        Args:
            path: Directory containing the tokenizer artifact.

        Returns:
            Loaded :class:`Tokenizer`.

        Raises:
            :class:`~abctokz.exceptions.SchemaVersionError`: If the artifact
                has an incompatible schema version.
            :class:`~abctokz.exceptions.SerializationError`: If required files
                are missing.
        """
        p = Path(path)
        manifest_path = p / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise SerializationError(f"No manifest found at {manifest_path}.")

        manifest_data = load_json(manifest_path)
        meta = ArtifactMetadata.from_dict(manifest_data)

        if meta.schema_version != SCHEMA_VERSION:
            raise SchemaVersionError(meta.schema_version, SCHEMA_VERSION)

        model_type = meta.model_type
        model: Model
        if model_type == "wordlevel":
            model = WordLevelModel.load(p)
            decoder: Decoder = WordDecoder()
        elif model_type == "bpe":
            model = BPEModel.load(p)
            decoder = SubwordDecoder()
        elif model_type == "unigram":
            model = UnigramModel.load(p)
            decoder = SubwordDecoder()
        else:
            raise SerializationError(f"Unknown model_type: {model_type!r}")

        # Try to restore full pipeline config if present
        normalizer = None
        pretokenizer = None
        post_processor = None
        tok_cfg: TokenizerConfig | None = None
        cfg_path = p / CONFIG_FILENAME
        if cfg_path.exists():
            cfg_data = load_json(cfg_path)
            raw_cfg = cfg_data.get("tokenizer_config") if isinstance(cfg_data, dict) else None
            if isinstance(raw_cfg, dict):
                try:
                    tok_cfg = TokenizerConfig(**raw_cfg)
                    shell = cls.from_config(tok_cfg)
                    normalizer = shell._normalizer
                    pretokenizer = shell._pretokenizer
                    post_processor = shell._post_processor
                except Exception:
                    tok_cfg = None

        # Load special tokens if present
        st_path = p / SPECIAL_TOKENS_FILENAME
        special_tokens: dict[str, SpecialToken] = {}
        if st_path.exists():
            st_data = load_json(st_path)
            special_tokens = {k: SpecialToken.from_dict(v) for k, v in st_data.items()}

        logger.info("Tokenizer loaded from %s (%s, vocab=%d)", path, model_type, meta.vocab_size)
        return cls(
            model=model,
            normalizer=normalizer,
            pretokenizer=pretokenizer,
            post_processor=post_processor,
            decoder=decoder,
            special_tokens=special_tokens,
            tokenizer_config=tok_cfg,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _infer_model_type(self) -> str:
        """Infer the model type string from the model instance."""
        if isinstance(self._model, WordLevelModel):
            return "wordlevel"
        if isinstance(self._model, BPEModel):
            return "bpe"
        if isinstance(self._model, UnigramModel):
            return "unigram"
        return "unknown"

    def __repr__(self) -> str:
        return (
            f"Tokenizer(model={self._infer_model_type()!r}, "
            f"vocab_size={self._model.get_vocab_size()})"
        )


class _PlaceholderModel(Model):
    """Temporary placeholder model before training is complete."""

    def tokenize(self, sequence: str) -> list[tuple[str, int]]:
        raise RuntimeError("Tokenizer has not been trained yet.")

    def save(self, directory: str | Path) -> None:
        pass

    @classmethod
    def load(cls, directory: str | Path) -> "_PlaceholderModel":
        return cls()

    def get_vocab(self) -> dict[str, int]:
        return {}


# Backward-compatibility alias
Tokenizer = AugenblickTokenizer
