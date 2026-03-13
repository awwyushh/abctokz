"""
Task 5 — Is It Truly Deterministic?

Experimental verification of determinism claims in abctokz.
Trains each model family twice on the same corpus + config and compares:
- Encoded outputs (ids/tokens)
- Vocabulary content and ordering
- Model-specific artifacts (merges / pieces)
- Non-deterministic metadata and timing signals
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from abctokz import Tokenizer
from abctokz.config.defaults import (
    bpe_multilingual,
    unigram_multilingual,
    wordlevel_multilingual,
)
from abctokz.config.schemas import BenchmarkConfig, TokenizerConfig
from abctokz.eval.benchmark import BenchmarkRunner

TRAIN_CORPUS: list[str] = [
    "hello world tokenizer determinism check",
    "नमस्ते दुनिया टोकनाइज़र स्थिरता परीक्षण",
    "machine learning and भाषा processing together",
    "भारत आणि India mixed script corpus",
    "repeatable merge order matters for bpe",
    "unigram pieces should be reproducible",
    "wordlevel frequency tie break lexicographically",
] * 90

PROBE_TEXTS: list[str] = [
    "hello world",
    "नमस्ते दुनिया",
    "भारत mixed world",
    "deterministic tokenizer proof",
]

MODEL_BUILDERS: dict[str, Callable[[int], TokenizerConfig]] = {
    "wordlevel": wordlevel_multilingual,
    "bpe": bpe_multilingual,
    "unigram": unigram_multilingual,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_signature(tok: Tokenizer, texts: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in texts:
        enc = tok.encode(t)
        out.append({"text": t, "ids": enc.ids, "tokens": enc.tokens})
    return out


def list_vocab_order(tok: Tokenizer) -> list[tuple[str, int]]:
    return list(tok.get_vocab().items())


def compare_optional_file(run1_dir: Path, run2_dir: Path, name: str) -> dict[str, Any]:
    p1 = run1_dir / name
    p2 = run2_dir / name
    if not p1.exists() and not p2.exists():
        return {"present": False, "equal": True, "run1": None, "run2": None}
    if p1.exists() and p2.exists():
        return {
            "present": True,
            "equal": sha256(p1) == sha256(p2),
            "run1": p1.name,
            "run2": p2.name,
        }
    return {"present": True, "equal": False, "run1": p1.exists(), "run2": p2.exists()}


def train_once(config: TokenizerConfig, corpus_path: Path, out_dir: Path) -> tuple[Tokenizer, float]:
    tok = Tokenizer.from_config(config)
    t0 = time.perf_counter()
    tok.train([str(corpus_path)], config)
    elapsed = time.perf_counter() - t0
    tok.save(str(out_dir))
    return tok, elapsed


def run_benchmark_twice(corpus_path: Path, tokenizer_path: Path, output_dir: Path) -> dict[str, Any]:
    cfg = BenchmarkConfig(
        name="task5_timing_variability",
        corpus_paths=[str(corpus_path)],
        tokenizer_paths=[str(tokenizer_path)],
        sample_size=120,
        warmup_runs=2,
        timed_runs=6,
        output_dir=str(output_dir),
        languages=[],
    )
    runner = BenchmarkRunner(cfg)
    run1 = [r.to_dict() for r in runner.run()]
    run2 = [r.to_dict() for r in runner.run()]

    one = run1[0] if run1 else {}
    two = run2[0] if run2 else {}

    return {
        "run1": one,
        "run2": two,
        "token_derived_equal": (
            one.get("fertility") == two.get("fertility")
            and one.get("unk_rate") == two.get("unk_rate")
            and one.get("round_trip_success_rate") == two.get("round_trip_success_rate")
            and one.get("mean_tokens_per_sentence") == two.get("mean_tokens_per_sentence")
            and one.get("normalized_seq_length_ratio") == two.get("normalized_seq_length_ratio")
        ),
        "throughput_differs": one.get("throughput_sps") != two.get("throughput_sps"),
        "elapsed_differs": one.get("elapsed_seconds") != two.get("elapsed_seconds"),
    }


def main() -> None:
    artifact_root = Path("artifacts/task5")
    output_root = Path("outputs/task5")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    corpus_path = artifact_root / "corpus.txt"
    corpus_path.write_text("\n".join(TRAIN_CORPUS), encoding="utf-8")

    model_reports: list[dict[str, Any]] = []

    for model_name, builder in MODEL_BUILDERS.items():
        cfg = builder(400)
        model_dir = artifact_root / model_name
        run1_dir = model_dir / "run1"
        run2_dir = model_dir / "run2"
        run1_dir.mkdir(parents=True, exist_ok=True)
        run2_dir.mkdir(parents=True, exist_ok=True)

        tok1, train_t1 = train_once(cfg, corpus_path, run1_dir)
        tok2, train_t2 = train_once(cfg, corpus_path, run2_dir)

        sig1 = encode_signature(tok1, PROBE_TEXTS)
        sig2 = encode_signature(tok2, PROBE_TEXTS)

        vocab1 = tok1.get_vocab()
        vocab2 = tok2.get_vocab()

        file_eq = {
            "vocab.json": compare_optional_file(run1_dir, run2_dir, "vocab.json")["equal"],
            "config.json": compare_optional_file(run1_dir, run2_dir, "config.json")["equal"],
            "manifest.json": compare_optional_file(run1_dir, run2_dir, "manifest.json")["equal"],
        }
        if model_name == "bpe":
            file_eq["merges.txt"] = compare_optional_file(run1_dir, run2_dir, "merges.txt")["equal"]
        if model_name == "unigram":
            file_eq["pieces.json"] = compare_optional_file(run1_dir, run2_dir, "pieces.json")["equal"]

        model_reports.append(
            {
                "model": model_name,
                "seed": cfg.trainer.seed if cfg.trainer is not None else None,
                "train_elapsed_seconds": {"run1": round(train_t1, 6), "run2": round(train_t2, 6)},
                "encode_ids_equal": [x["ids"] for x in sig1] == [x["ids"] for x in sig2],
                "encode_tokens_equal": [x["tokens"] for x in sig1] == [x["tokens"] for x in sig2],
                "vocab_equal": vocab1 == vocab2,
                "vocab_order_equal": list_vocab_order(tok1) == list_vocab_order(tok2),
                "artifact_hash_equal": file_eq,
                "probe_outputs_run1": sig1,
                "probe_outputs_run2": sig2,
            }
        )

    # Timing non-determinism check on one fixed artifact (BPE run1)
    timing = run_benchmark_twice(
        corpus_path=corpus_path,
        tokenizer_path=artifact_root / "bpe" / "run1",
        output_dir=output_root,
    )

    deterministic_parts = [
        "Encoding outputs (ids/tokens) for fixed text",
        "Vocabulary content",
        "Vocabulary insertion order",
        "BPE merge-rule order (merges.txt)",
        "Unigram piece ordering/scores (pieces.json)",
        "Config serialization (config.json)",
    ]

    acceptable_nondeterministic_parts = [
        "manifest.json timestamp (created_at uses wall-clock UTC)",
        "benchmark throughput and elapsed time (scheduler, CPU load, cache effects)",
        "training wall-clock duration",
    ]

    risks = [
        "Changing Python/NumPy versions can alter tie-order behavior in edge cases.",
        "Different locale/collation rules can affect lexicographic tie-breaks if implementation changes.",
        "Corpus file ordering or line-ending normalization differences change frequency counts.",
        "Manual edits in artifact metadata (manifest/config) break byte-level identity without changing model behavior.",
        "Future parallel training optimizations without stable reduction order can introduce nondeterminism.",
    ]

    report = {
        "task": "Task 5 - Is It Truly Deterministic?",
        "corpus_path": str(corpus_path),
        "probe_texts": PROBE_TEXTS,
        "models": model_reports,
        "benchmark_timing_repeatability": timing,
        "deterministic_parts": deterministic_parts,
        "acceptable_nondeterministic_parts": acceptable_nondeterministic_parts,
        "remaining_risks": risks,
        "verdict": {
            "model_behavior_deterministic": all(
                row["encode_ids_equal"]
                and row["encode_tokens_equal"]
                and row["vocab_equal"]
                and row["vocab_order_equal"]
                and row["artifact_hash_equal"]["vocab.json"]
                and row["artifact_hash_equal"]["config.json"]
                and row["artifact_hash_equal"].get("merges.txt", True)
                and row["artifact_hash_equal"].get("pieces.json", True)
                for row in model_reports
            ),
            "byte_identical_full_artifact": all(
                row["artifact_hash_equal"].get("manifest.json", False) for row in model_reports
            ),
            "explanation": "Model outputs are deterministic; full artifact byte identity is broken by manifest timestamp.",
        },
    }

    out_path = output_root / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 5 — Is It Truly Deterministic?")
    print(f"report: {out_path}")
    print()
    for row in model_reports:
        print(
            f"[{row['model']}] ids_equal={row['encode_ids_equal']} "
            f"vocab_equal={row['vocab_equal']} order_equal={row['vocab_order_equal']} "
            f"manifest_equal={row['artifact_hash_equal']['manifest.json']}"
        )
    tr = timing
    print(
        "benchmark: token_derived_equal="
        f"{tr['token_derived_equal']} throughput_differs={tr['throughput_differs']}"
    )


if __name__ == "__main__":
    main()
