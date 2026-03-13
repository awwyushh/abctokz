from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
from abctokz.config.schemas import TokenizerConfig

MODEL_BUILDERS: dict[str, Callable[[int], TokenizerConfig]] = {
    "wordlevel": wordlevel_multilingual,
    "bpe": bpe_multilingual,
    "unigram": unigram_multilingual,
}

ENGLISH_ONLY_CORPUS: list[str] = [
    "the quick brown fox jumps over the lazy dog",
    "tokenization quality matters for language models",
    "this corpus is intentionally english only",
    "unknown words should reveal model behavior",
    "we test robustness and fallback behavior",
] * 80

TEST_INPUTS: list[tuple[str, str]] = [
    ("devanagari_on_english_model", "नमस्ते भारत"),
    ("emoji_mixed", "hello 😊 world"),
    ("rare_english_word", "electroencephalographically"),
    ("currency_symbol", "price is €100"),
]


def choose_runtime_tokenizer(trained: Tokenizer, loaded: Tokenizer, probe_text: str) -> tuple[Tokenizer, str]:
    trained_enc = trained.encode(probe_text)
    loaded_enc = loaded.encode(probe_text)
    if trained_enc.tokens == loaded_enc.tokens and trained_enc.ids == loaded_enc.ids:
        return loaded, "loaded"
    return trained, "trained"


def build_tokenizer(model_name: str, vocab_size: int, artifact_root: Path) -> tuple[Tokenizer, str]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    corpus_path = artifact_root / "english_only_corpus.txt"
    corpus_path.write_text("\n".join(ENGLISH_ONLY_CORPUS), encoding="utf-8")

    config: TokenizerConfig = MODEL_BUILDERS[model_name](vocab_size=vocab_size)
    trained = Tokenizer.from_config(config)
    trained.train([str(corpus_path)], config)

    model_dir = artifact_root / model_name
    trained.save(str(model_dir))
    loaded = Tokenizer.load(str(model_dir))
    return choose_runtime_tokenizer(trained, loaded, probe_text="hello world")


def unk_id(tokenizer: Tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    return vocab.get("<unk>", 0)


def run_case(tokenizer: Tokenizer, case_label: str, text: str) -> dict[str, object]:
    enc = tokenizer.encode(text)
    u = unk_id(tokenizer)
    unk_count = sum(1 for i in enc.ids if i == u)
    return {
        "case": case_label,
        "text": text,
        "word_count": len(text.split()),
        "token_count": len(enc.ids),
        "unk_count": unk_count,
        "unk_rate": (unk_count / max(len(enc.ids), 1)),
        "tokens": enc.tokens,
        "ids": enc.ids,
        "decoded": tokenizer.decode(enc.ids),
    }


def classify_gracefulness(results: dict[str, list[dict[str, object]]]) -> dict[str, str]:
    score: dict[str, float] = {}
    for model_name, rows in results.items():
        unk_total = sum(float(r["unk_count"]) for r in rows)
        tok_total = sum(float(r["token_count"]) for r in rows)
        score[model_name] = unk_total / max(tok_total, 1.0)

    ordered = sorted(score.items(), key=lambda x: x[1])
    graceful = ordered[0][0]
    fragile = ordered[-1][0]
    return {
        "most_graceful": graceful,
        "most_fragile": fragile,
        "unk_ratio_by_model": {k: round(v, 4) for k, v in score.items()},
    }


def main() -> None:
    artifact_root = Path("artifacts/task6")
    output_path = Path("outputs/task6/report.json")

    runtime_sources: dict[str, str] = {}
    models: dict[str, Tokenizer] = {}
    for model_name in ["wordlevel", "bpe", "unigram"]:
        tok, source = build_tokenizer(model_name, vocab_size=200, artifact_root=artifact_root)
        models[model_name] = tok
        runtime_sources[model_name] = source

    results: dict[str, list[dict[str, object]]] = {}
    for model_name, tok in models.items():
        rows: list[dict[str, object]] = []
        for label, text in TEST_INPUTS:
            rows.append(run_case(tok, label, text))
        results[model_name] = rows

    summary = classify_gracefulness(results)

    report = {
        "corpus": {
            "type": "english_only",
            "n_lines": len(ENGLISH_ONLY_CORPUS),
        },
        "runtime_sources": runtime_sources,
        "results": results,
        "summary": summary,
        "suggestion_without_retraining": (
            "Use a fallback routing strategy: if UNK rate on an input exceeds a threshold, "
            "re-encode that input with a more robust external tokenizer (e.g., SentencePiece/BPE baseline) "
            "or a second in-project tokenizer better matched to that script."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 6: Making the Tokenizer Say <unk>")
    print(f"report: {output_path}")
    print()
    for model_name in ["wordlevel", "bpe", "unigram"]:
        print(f"[{model_name}] source={runtime_sources[model_name]}")
        for row in results[model_name]:
            print(
                f"  {row['case']}: tokens={row['token_count']} unk={row['unk_count']} "
                f"unk_rate={row['unk_rate']:.3f}"
            )
        print()

    print("summary")
    print(f"  most_graceful: {summary['most_graceful']}")
    print(f"  most_fragile : {summary['most_fragile']}")
    print(f"  unk_ratio_by_model: {summary['unk_ratio_by_model']}")


if __name__ == "__main__":
    main()
