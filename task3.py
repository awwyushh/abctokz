from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual, wordlevel_multilingual
from abctokz.config.schemas import TokenizerConfig
from abctokz.eval.metrics import fertility

MODEL_BUILDERS: dict[str, Callable[[int], TokenizerConfig]] = {
    "wordlevel": wordlevel_multilingual,
    "bpe": bpe_multilingual,
    "unigram": unigram_multilingual,
}

ANTHEM_TRANSLIT = (
    "Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata "
    "Punjab Sindhu Gujarat Maratha Dravida Utkala Banga "
    "Vindhya Himachala Yamuna Ganga Uchchhala Jaladhi Taranga "
    "Tava Shubha Name Jage Tava Shubha Ashisa Mage "
    "Gahe Tava Jaya Gatha "
    "Jana Gana Mangala Dayaka Jaya He Bharata Bhagya Vidhata "
    "Jaya He Jaya He Jaya He "
    "Jaya Jaya Jaya Jaya He"
)

ANTHEM_DEVANAGARI = (
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता "
    "पंजाब सिंधु गुजरात मराठा द्रविड़ उत्कल बंग "
    "विंध्य हिमाचल यमुना गंगा उच्छल जलधि तरंग "
    "तव शुभ नामे जागे तव शुभ आशीष मांगे "
    "गाहे तव जय गाथा "
    "जन गण मंगलदायक जय हे भारत भाग्य विधाता "
    "जय हे जय हे जय हे "
    "जय जय जय जय हे"
)

DEFAULT_CORPUS_LINES: list[str] = [
    ANTHEM_TRANSLIT,
    ANTHEM_DEVANAGARI,
    "Vande Mataram",
    "सारे जहाँ से अच्छा हिन्दोस्तां हमारा",
    "भारत माता की जय",
    "Jana Gana Mana",
    "जन गण मन",
] * 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="task3", description="Run Task 3 anthem script comparison")
    parser.add_argument("--model", choices=["wordlevel", "bpe", "unigram"], default="bpe")
    parser.add_argument("--corpus", nargs="*", default=[], help="Optional custom corpus file(s)")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--artifact-dir", default="artifacts/task3")
    parser.add_argument("--output-json", default="outputs/task3/report.json")
    return parser.parse_args()


def read_lines(paths: list[str]) -> list[str]:
    if not paths:
        return list(DEFAULT_CORPUS_LINES)
    lines: list[str] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        lines.extend(line.strip() for line in text.splitlines() if line.strip())
    return lines


def choose_runtime_tokenizer(trained: Tokenizer, loaded: Tokenizer, probe_text: str) -> tuple[Tokenizer, str]:
    trained_enc = trained.encode(probe_text)
    loaded_enc = loaded.encode(probe_text)
    if trained_enc.tokens == loaded_enc.tokens and trained_enc.ids == loaded_enc.ids:
        return loaded, "loaded"
    return trained, "trained"


def build_tokenizer(model_name: str, vocab_size: int, corpus_lines: list[str], artifact_dir: Path) -> tuple[Tokenizer, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = artifact_dir / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

    config = MODEL_BUILDERS[model_name](vocab_size=vocab_size)
    trained = Tokenizer.from_config(config)
    trained.train([str(corpus_path)], config)
    trained.save(str(artifact_dir / model_name))
    loaded = Tokenizer.load(str(artifact_dir / model_name))
    return choose_runtime_tokenizer(trained, loaded, probe_text=ANTHEM_TRANSLIT)


def text_stats(tokenizer: Tokenizer, label: str, text: str) -> dict[str, object]:
    enc = tokenizer.encode(text)
    word_count = len(text.split())
    token_count = len(enc.ids)
    fert = fertility([enc], [word_count])
    return {
        "label": label,
        "word_count": word_count,
        "token_count": token_count,
        "fertility": fert,
        "tokens": enc.tokens,
        "ids": enc.ids,
        "decoded": tokenizer.decode(enc.ids),
    }


def tiktoken_stats(texts: dict[str, str]) -> dict[str, object]:
    try:
        import tiktoken  # type: ignore[import-untyped]
    except Exception:
        return {"available": False, "reason": "tiktoken not installed"}

    enc = tiktoken.get_encoding("cl100k_base")
    out: dict[str, object] = {"available": True, "encoding": "cl100k_base", "results": {}}
    for label, text in texts.items():
        ids = enc.encode(text)
        out["results"][label] = {
            "word_count": len(text.split()),
            "token_count": len(ids),
            "fertility": (len(ids) / max(len(text.split()), 1)),
        }
    return out


def explanation(abctokz_results: dict[str, dict[str, object]]) -> str:
    tr = abctokz_results["transliteration"]
    dv = abctokz_results["devanagari"]
    tr_tokens = int(tr["token_count"])
    dv_tokens = int(dv["token_count"])
    tr_f = float(tr["fertility"])
    dv_f = float(dv["fertility"])

    if dv_tokens > tr_tokens:
        trend = "Devanagari produced more tokens than transliteration"
    elif dv_tokens < tr_tokens:
        trend = "Transliteration produced more tokens than Devanagari"
    else:
        trend = "Both scripts produced the same token count"

    return (
        f"{trend}. "
        "The difference is mainly caused by script-level symbol distribution and how the learned vocabulary "
        "covers frequent units from the training corpus. It is not script alone: corpus composition and model "
        "family strongly influence whether words remain whole or split into subword pieces. "
        f"Observed fertility: transliteration={tr_f:.3f}, devanagari={dv_f:.3f}."
    )


def print_report(model_name: str, source: str, abctokz_results: dict[str, dict[str, object]], tiktok: dict[str, object], note: str) -> None:
    print("Task 3: The National Anthem Test")
    print()
    print(f"model: {model_name}")
    print(f"tokenizer source used: {source}")
    print()

    for key in ["transliteration", "devanagari"]:
        r = abctokz_results[key]
        print(f"[{key}]")
        print(f"  words     : {r['word_count']}")
        print(f"  tokens    : {r['token_count']}")
        print(f"  fertility : {r['fertility']:.3f}")
        print(f"  sample tokens: {r['tokens'][:25]}")
        print()

    print("explanation")
    print(f"  {note}")
    print()

    print("bonus: tiktoken comparison")
    if not tiktok.get("available"):
        print(f"  unavailable: {tiktok.get('reason')}")
    else:
        print(f"  encoding: {tiktok['encoding']}")
        results = tiktok["results"]
        for key in ["transliteration", "devanagari"]:
            r = results[key]
            print(f"  [{key}] words={r['word_count']} tokens={r['token_count']} fertility={r['fertility']:.3f}")


def main() -> None:
    args = parse_args()
    corpus_lines = read_lines(args.corpus)
    tokenizer, source = build_tokenizer(
        model_name=args.model,
        vocab_size=args.vocab_size,
        corpus_lines=corpus_lines,
        artifact_dir=Path(args.artifact_dir),
    )

    abctokz_results = {
        "transliteration": text_stats(tokenizer, "transliteration", ANTHEM_TRANSLIT),
        "devanagari": text_stats(tokenizer, "devanagari", ANTHEM_DEVANAGARI),
    }

    tiktok = tiktoken_stats({
        "transliteration": ANTHEM_TRANSLIT,
        "devanagari": ANTHEM_DEVANAGARI,
    })

    note = explanation(abctokz_results)

    output = {
        "model": args.model,
        "vocab_size": args.vocab_size,
        "tokenizer_source": source,
        "abctokz": abctokz_results,
        "tiktoken": tiktok,
        "explanation": note,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(args.model, source, abctokz_results, tiktok, note)
    print()
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
