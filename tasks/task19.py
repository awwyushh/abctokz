from __future__ import annotations

import argparse
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

TASK19_INPUTS: list[tuple[str, str]] = [
    ("easy_english", "hello world"),
    ("complex_english", "internationalization is nontrivial"),
    ("simple_hindi", "नमस्ते दुनिया"),
    ("complex_hindi", "प्रौद्योगिकीकरण महत्वपूर्ण है"),
    ("mixed_script", "नमस्ते world 2026"),
]

DEFAULT_CORPUS_LINES: list[str] = [
    "hello world",
    "internationalization is nontrivial",
    "tokenization helps multilingual models",
    "the tokenizer should handle unseen morphology",
    "नमस्ते दुनिया",
    "यह एक सरल वाक्य है",
    "प्रौद्योगिकीकरण महत्वपूर्ण है",
    "मशीन लर्निंग के लिए टोकनाइज़ेशन उपयोगी है",
    "नमस्ते world 2026",
    "हिन्दी और English का मिश्रण सामान्य है",
] * 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="task19", description="Run Task 19 model comparison")
    parser.add_argument("--corpus", nargs="*", default=[], help="One or more corpus text files")
    parser.add_argument("--vocab-size", type=int, default=200, help="Vocabulary size used for all models")
    parser.add_argument("--artifact-root", default="artifacts/task19", help="Where model artifacts are saved")
    parser.add_argument("--output-json", default="outputs/task19/report.json", help="Path to save structured report")
    return parser.parse_args()


def read_lines(paths: list[str]) -> list[str]:
    if not paths:
        return list(DEFAULT_CORPUS_LINES)
    lines: list[str] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        lines.extend(line.strip() for line in text.splitlines() if line.strip())
    return lines


def choose_runtime_tokenizer(
    trained: Tokenizer,
    loaded: Tokenizer,
    probe_text: str,
) -> tuple[Tokenizer, str]:
    trained_encoding = trained.encode(probe_text)
    loaded_encoding = loaded.encode(probe_text)
    if trained_encoding.tokens == loaded_encoding.tokens and trained_encoding.ids == loaded_encoding.ids:
        return loaded, "loaded"
    return trained, "trained"


def train_all_models(
    corpus_lines: list[str],
    vocab_size: int,
    artifact_root: Path,
) -> tuple[dict[str, Tokenizer], dict[str, str]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    corpus_path = artifact_root / "shared_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

    tokenizers: dict[str, Tokenizer] = {}
    tokenizer_sources: dict[str, str] = {}
    probe_text = corpus_lines[0]
    for model_name, build_config in MODEL_BUILDERS.items():
        config = build_config(vocab_size=vocab_size)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)
        model_dir = artifact_root / model_name
        tokenizer.save(str(model_dir))
        loaded = Tokenizer.load(str(model_dir))
        runtime_tokenizer, source = choose_runtime_tokenizer(tokenizer, loaded, probe_text)
        tokenizers[model_name] = runtime_tokenizer
        tokenizer_sources[model_name] = source
    return tokenizers, tokenizer_sources


def normalize_token(token: str) -> str:
    if token.startswith("##"):
        return token[2:]
    return token


def vocab_profile(vocab: dict[str, int], corpus_words: set[str]) -> dict[str, int | float | str]:
    whole_word = 0
    subword = 0
    character = 0
    special = 0

    for token in vocab:
        if token.startswith("<") and token.endswith(">"):
            special += 1
            continue
        clean = normalize_token(token)
        if clean in corpus_words:
            whole_word += 1
        elif len(clean) <= 1:
            character += 1
        else:
            subword += 1

    total = len(vocab)
    dominant = max(
        [("whole_word", whole_word), ("subword", subword), ("character", character)],
        key=lambda x: x[1],
    )[0]
    return {
        "total": total,
        "whole_word": whole_word,
        "subword": subword,
        "character": character,
        "special": special,
        "dominant": dominant,
    }


def side_by_side(tokenizers: dict[str, Tokenizer], inputs: list[tuple[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, text in inputs:
        row: dict[str, object] = {"label": label, "text": text, "encodings": {}}
        for model_name, tokenizer in tokenizers.items():
            enc = tokenizer.encode(text)
            row["encodings"][model_name] = {
                "tokens": enc.tokens,
                "ids": enc.ids,
                "decoded": tokenizer.decode(enc.ids),
            }
        rows.append(row)
    return rows


def print_report(
    rows: list[dict[str, object]],
    profiles: dict[str, dict[str, int | float | str]],
    tokenizer_sources: dict[str, str],
) -> None:
    print("Task 19: BPE vs Unigram vs WordLevel")
    print()
    for row in rows:
        label = row["label"]
        text = row["text"]
        print(f"[{label}] {text}")
        encodings = row["encodings"]
        for model_name in ["wordlevel", "bpe", "unigram"]:
            info = encodings[model_name]
            print(f"  {model_name:10} tokens: {info['tokens']}")
            print(f"  {model_name:10} ids   : {info['ids']}")
        print()

    print("Vocabulary dominance")
    for model_name in ["wordlevel", "bpe", "unigram"]:
        p = profiles[model_name]
        print(
            f"  {model_name:10} dominant={p['dominant']} total={p['total']} "
            f"whole_word={p['whole_word']} subword={p['subword']} "
            f"character={p['character']} special={p['special']}"
        )

    print()
    print("Tokenizer source used for analysis")
    for model_name in ["wordlevel", "bpe", "unigram"]:
        print(f"  {model_name:10} source={tokenizer_sources[model_name]}")

    print()
    print("Recommended choices")
    print("  low_resource_language: unigram")
    print("  agglutinative_language: unigram (bpe as close second)")
    print("  consistent_boundaries_across_languages: wordlevel")
    print()
    print("Segmentation assumptions")
    print("  wordlevel: words are atomic; unseen words map to <unk>")
    print("  bpe: frequent adjacent units are merged into reusable subwords")
    print("  unigram: best-probability piece sequence explains each token")


def main() -> None:
    args = parse_args()
    corpus_lines = read_lines(args.corpus)
    tokenizers, tokenizer_sources = train_all_models(
        corpus_lines=corpus_lines,
        vocab_size=args.vocab_size,
        artifact_root=Path(args.artifact_root),
    )
    corpus_words = {w for line in corpus_lines for w in line.split()}
    profiles = {
        model_name: vocab_profile(tokenizer.get_vocab(), corpus_words)
        for model_name, tokenizer in tokenizers.items()
    }
    rows = side_by_side(tokenizers, TASK19_INPUTS)

    output = {
        "inputs": rows,
        "vocabulary_profiles": profiles,
        "recommendations": {
            "low_resource_language": "unigram",
            "agglutinative_language": "unigram",
            "consistent_boundaries_across_languages": "wordlevel",
        },
        "tokenizer_sources": tokenizer_sources,
        "segmentation_assumptions": {
            "wordlevel": "words are atomic; unseen words map to <unk>",
            "bpe": "frequent adjacent units are merged into reusable subwords",
            "unigram": "best-probability piece sequence explains each token",
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(rows, profiles, tokenizer_sources)
    print()
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
