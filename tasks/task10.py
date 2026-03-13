"""
Task 10 — The Compression Trade-off

Tokenizer "compression" means producing fewer tokens for the same text (lower fertility).
This script finds a configuration change that improves one metric while making another worse,
and reports the tension between them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from abctokz import Tokenizer
from abctokz.config.defaults import multilingual_shared_normalizer
from abctokz.config.schemas import (
    BPEConfig,
    BPETrainerConfig,
    SequencePreTokenizerConfig,
    TokenizerConfig,
)
from abctokz.config.schemas import DevanagariAwarePreTokenizerConfig
from abctokz.eval.metrics import (
    fertility,
    round_trip_success_rate,
    unk_rate,
)
from abctokz.types import Encoding

# Corpus for vocab_size trade-off: many unique English sentences so BPE
# can fill 1k vs 8k vocab (artifact size and fertility differ).
TRAIN_CORPUS_VOCAB: list[str] = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning and natural language processing",
    "tokenization is useful for NLP applications",
    "subword tokenizers reduce vocabulary size",
    "byte pair encoding merges frequent pairs",
    "attention mechanisms in transformer models",
    "internationalization and localization matter",
    "rare words and unknown tokens appear",
    "vocabulary size affects compression ratio",
    "fertility measures tokens per word",
    "language models predict next tokens",
    "embedding layers map tokens to vectors",
    "transformer architecture uses self attention",
    "preprocessing pipelines normalize text",
    "corpus statistics drive merge decisions",
    "decoding reconstructs text from indices",
    "vocabulary files store token strings",
    "merge rules define subword composition",
    "alphabet size limits initial symbols",
    "frequency counts determine merge order",
] * 80

# Corpus for min_frequency: mixed so rare vs frequent words matter
TRAIN_CORPUS_MF: list[str] = [
    "hello world",
    "नमस्ते दुनिया",
    "tokenization is useful for NLP",
    "भारत एक विशाल देश है",
    "machine learning and natural language processing",
    "हिन्दी भाषा में टोकनाइजेशन",
    "the quick brown fox jumps over the lazy dog",
    "गणपती बप्पा मोरया",
    "subword tokenizers reduce vocabulary size",
    "देवनागरी लिपि महत्वपूर्ण है",
    "internationalization and localization",
    "प्रौद्योगिकीकरण महत्वपूर्ण है",
    "attention mechanisms in transformers",
    "सारे जहाँ से अच्छा हिन्दोस्तां",
    "byte pair encoding algorithm",
    "मराठी भाषेत टोकनायझेशन",
    "vocabulary size affects fertility",
    "झूलेलाल सिंधी समाज",
    "rare words and unknown tokens",
    "उच्चारण और लेखन में अंतर",
] * 60

# Test sentences: mixed (for min_frequency and general reporting)
TEST_SENTENCES: list[str] = [
    "hello world",
    "नमस्ते दुनिया",
    "tokenization is useful",
    "भारत विशाल देश है",
    "machine learning",
    "हिन्दी भाषा",
    "quick brown fox",
    "गणपती बप्पा",
    "subword tokenizers",
    "देवनागरी लिपि",
]

# English-only test set for vocab_size experiment (train corpus is English-only)
TEST_SENTENCES_EN: list[str] = [
    "hello world",
    "tokenization is useful",
    "machine learning",
    "quick brown fox",
    "subword tokenizers",
    "byte pair encoding",
    "vocabulary size",
    "natural language processing",
    "attention mechanisms",
    "internationalization",
]


def build_bpe_config(
    vocab_size: int = 2000,
    min_frequency: int = 2,
    limit_alphabet: int | None = None,
) -> TokenizerConfig:
    return TokenizerConfig(
        normalizer=multilingual_shared_normalizer(),
        pretokenizer=SequencePreTokenizerConfig(
            pretokenizers=[
                DevanagariAwarePreTokenizerConfig(
                    split_on_whitespace=True,
                    split_on_script_boundary=True,
                ),
            ]
        ),
        model=BPEConfig(unk_token="<unk>", vocab_size=vocab_size),
        trainer=BPETrainerConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            special_tokens=["<unk>"],
        ),
    )


def train_and_evaluate(
    config: TokenizerConfig,
    corpus_path: Path,
    artifact_dir: Path,
    config_label: str,
    test_sentences: list[str] | None = None,
) -> dict[str, Any]:
    """Train tokenizer, save artifact, run metrics. Uses in-memory tokenizer for eval."""
    sentences = test_sentences if test_sentences is not None else TEST_SENTENCES
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([str(corpus_path)], config)

    artifact_path = artifact_dir / config_label.replace(" ", "_")
    artifact_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(artifact_path))

    # Evaluate using in-memory tokenizer (avoids load-path drift)
    encodings: list[Encoding] = tokenizer.encode_batch(sentences)
    ref_counts = [len(s.split()) for s in sentences]
    decoded = [tokenizer.decode(enc.ids) for enc in encodings]

    # Throughput: encode many times
    n_warmup, n_runs = 3, 20
    for _ in range(n_warmup):
        tokenizer.encode_batch(sentences)
    start = time.perf_counter()
    for _ in range(n_runs):
        tokenizer.encode_batch(sentences)
    elapsed = time.perf_counter() - start
    throughput_sps = (len(sentences) * n_runs) / max(elapsed, 1e-9)

    # Artifact size (directory)
    artifact_size_bytes = sum(
        f.stat().st_size for f in artifact_path.rglob("*") if f.is_file()
    )

    trainer = config.trainer
    return {
        "config_label": config_label,
        "vocab_size": config.model.vocab_size,
        "min_frequency": getattr(trainer, "min_frequency", None) if trainer else None,
        "limit_alphabet": getattr(trainer, "limit_alphabet", None) if trainer else None,
        "fertility": round(fertility(encodings, ref_counts), 4),
        "unk_rate": round(unk_rate(encodings, unk_id=0), 6),
        "round_trip_success_rate": round(
            round_trip_success_rate(TEST_SENTENCES, decoded), 6
        ),
        "mean_tokens_per_sentence": round(
            sum(len(e) for e in encodings) / len(encodings), 2
        ),
        "throughput_sps": round(throughput_sps, 1),
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_size_kib": round(artifact_size_bytes / 1024, 2),
    }


def main() -> None:
    artifact_root = Path("artifacts/task10")
    output_path = Path("outputs/task10/report.json")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_vocab_path = artifact_root / "corpus_vocab.txt"
    corpus_mf_path = artifact_root / "corpus_minfreq.txt"
    corpus_vocab_path.write_text("\n".join(TRAIN_CORPUS_VOCAB), encoding="utf-8")
    corpus_mf_path.write_text("\n".join(TRAIN_CORPUS_MF), encoding="utf-8")

    results: dict[str, Any] = {
        "test_sentences_count": len(TEST_SENTENCES),
        "trade_off_vocab_size": [],
        "trade_off_min_frequency": [],
        "trade_off_limit_alphabet": [],
        "summary": {},
    }

    # ---- Trade-off 1: vocab_size (compression vs artifact size / throughput) ----
    print("Task 10 — The Compression Trade-off")
    print("=" * 60)
    print("\n1. Varying vocab_size (min_frequency=2) [English-heavy corpus]")
    print("-" * 40)

    for vs in [1000, 3000, 8000]:
        config = build_bpe_config(vocab_size=vs, min_frequency=2)
        label = f"vocab_{vs}"
        row = train_and_evaluate(
            config,
            corpus_vocab_path,
            artifact_root,
            label,
            test_sentences=TEST_SENTENCES_EN,
        )
        results["trade_off_vocab_size"].append(row)
        print(
            f"  vocab_size={vs:5d} | fertility={row['fertility']:.4f} | "
            f"unk_rate={row['unk_rate']:.4f} | round_trip={row['round_trip_success_rate']:.4f} | "
            f"artifact={row['artifact_size_kib']:.1f} KiB | throughput={row['throughput_sps']:.0f} sps"
        )

    # ---- Trade-off 2: min_frequency (compression vs robustness) ----
    print("\n2. Varying min_frequency (vocab_size=2000)")
    print("-" * 40)

    for mf in [1, 2, 5]:
        config = build_bpe_config(vocab_size=2000, min_frequency=mf)
        label = f"minfreq_{mf}"
        row = train_and_evaluate(config, corpus_mf_path, artifact_root, label)
        results["trade_off_min_frequency"].append(row)
        print(
            f"  min_frequency={mf} | fertility={row['fertility']:.4f} | "
            f"unk_rate={row['unk_rate']:.4f} | round_trip={row['round_trip_success_rate']:.4f}"
        )

    # ---- Trade-off 3: limit_alphabet (coverage vs artifact size) ----
    print("\n3. Varying limit_alphabet (vocab_size=2000, min_frequency=2) [mixed corpus]")
    print("-" * 40)

    for limit in [60, 120, None]:
        config = build_bpe_config(
            vocab_size=2000, min_frequency=2, limit_alphabet=limit
        )
        label = f"limit_alph_{limit}" if limit is not None else "limit_alph_None"
        row = train_and_evaluate(
            config, corpus_mf_path, artifact_root, label,
            test_sentences=TEST_SENTENCES,
        )
        results["trade_off_limit_alphabet"].append(row)
        print(
            f"  limit_alphabet={limit!s:4} | fertility={row['fertility']:.4f} | "
            f"unk_rate={row['unk_rate']:.4f} | round_trip={row['round_trip_success_rate']:.4f} | "
            f"artifact={row['artifact_size_kib']:.1f} KiB"
        )

    # Summary for writeup
    v_rows = results["trade_off_vocab_size"]
    m_rows = results["trade_off_min_frequency"]
    la_rows = results["trade_off_limit_alphabet"]
    results["summary"] = {
        "vocab_size_trade_off": (
            "Larger vocab_size improves compression (lower fertility) but increases "
            "artifact size and can reduce encode throughput."
        ),
        "min_frequency_trade_off": (
            "Lower min_frequency keeps more rare words in training, improving "
            "compression (lower fertility); higher min_frequency can improve "
            "round-trip or robustness by avoiding spurious rare-word merges."
        ),
        "limit_alphabet_trade_off": (
            "Smaller limit_alphabet yields smaller artifact and faster load, but "
            "drops rare characters from the initial alphabet, increasing UNK rate "
            "and hurting round-trip on scripts that use those characters."
        ),
        "best_fertility_vocab": min(v_rows, key=lambda r: r["fertility"])["config_label"],
        "smallest_artifact_vocab": min(
            v_rows, key=lambda r: r["artifact_size_bytes"]
        )["config_label"],
        "best_fertility_minfreq": min(
            m_rows, key=lambda r: r["fertility"]
        )["config_label"],
        "limit_alphabet_unk_vs_artifact": (
            "Lower limit_alphabet → smaller artifact, higher UNK; "
            "higher/None → larger artifact, lower UNK."
        ),
    }

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
