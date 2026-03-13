from __future__ import annotations

import json
from pathlib import Path

from abctokz.config.defaults import bpe_multilingual
from abctokz.config.schemas import BenchmarkConfig, TokenizerConfig
from abctokz.eval.benchmark import BenchmarkRunner
from abctokz.tokenizer import Tokenizer

CORPUS = [
    "hello world",
    "नमस्ते दुनिया",
    "हिन्दी भाषा में टोकनाइजेशन",
    "BPE and Unigram are different",
] * 50


def setup_artifact_and_corpus(artifact_dir: Path, corpus_path: Path) -> Path:
    corpus_path.write_text("\n".join(CORPUS), encoding="utf-8")
    cfg: TokenizerConfig = bpe_multilingual(vocab_size=200)
    tok = Tokenizer.from_config(cfg)
    tok.train([str(corpus_path)], cfg)
    tok_dir = artifact_dir / "bpe_task11"
    tok.save(str(tok_dir))
    return tok_dir


def result_key(r: dict) -> tuple[str, str]:
    return (r["tokenizer_name"], r["language"] or "mixed")


def compare_runs(run1: list[dict], run2: list[dict]) -> dict:
    by_key1 = {result_key(r): r for r in run1}
    by_key2 = {result_key(r): r for r in run2}
    stable_fields = [
        "n_sentences",
        "mean_tokens_per_sentence",
        "fertility",
        "unk_rate",
        "round_trip_success_rate",
        "normalized_seq_length_ratio",
    ]
    comparison = {"stable_ok": True, "variable_differ": False, "per_run": []}
    for k in sorted(by_key1.keys()):
        if k not in by_key2:
            comparison["stable_ok"] = False
            comparison["per_run"].append({"key": list(k), "issue": "missing_in_run2"})
            continue
        a, b = by_key1[k], by_key2[k]
        for f in stable_fields:
            if a.get(f) != b.get(f):
                comparison["stable_ok"] = False
        if a.get("throughput_sps") != b.get("throughput_sps"):
            comparison["variable_differ"] = True
        comparison["per_run"].append({
            "tokenizer": k[0],
            "lang": k[1],
            "run1_throughput_sps": a.get("throughput_sps"),
            "run2_throughput_sps": b.get("throughput_sps"),
            "run1_elapsed": a.get("elapsed_seconds"),
            "run2_elapsed": b.get("elapsed_seconds"),
            "fertility_run1": a.get("fertility"),
            "fertility_run2": b.get("fertility"),
        })
    return comparison


def main() -> None:
    base = Path("artifacts/task11")
    out_dir = Path("outputs/task11")
    base.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = base / "corpus.txt"
    tok_path = setup_artifact_and_corpus(base, corpus_path)

    bench_cfg = BenchmarkConfig(
        name="task11_trust",
        corpus_paths=[str(corpus_path)],
        tokenizer_paths=[str(tok_path)],
        sample_size=80,
        warmup_runs=2,
        timed_runs=5,
        output_dir=str(out_dir),
        languages=[],
    )
    runner = BenchmarkRunner(bench_cfg)

    run1 = [r.to_dict() for r in runner.run()]
    run2 = [r.to_dict() for r in runner.run()]

    cmp = compare_runs(run1, run2)
    report = {
        "run1_results": run1,
        "run2_results": run2,
        "comparison": cmp,
        "takeaway": {
            "token_derived_metrics_stable": cmp["stable_ok"],
            "timing_metrics_vary": cmp["variable_differ"],
        },
    }
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Task 11: Can You Trust the Benchmark Numbers?")
    print(f"Ran same benchmark twice (same config, same artifact).")
    print(f"Stable (token-derived): {cmp['stable_ok']}")
    print(f"Timing differed: {cmp['variable_differ']}")
    if run1 and run2:
        r1, r2 = run1[0], run2[0]
        print(f"  Run1 throughput={r1.get('throughput_sps')} sps, Run2 throughput={r2.get('throughput_sps')} sps")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
