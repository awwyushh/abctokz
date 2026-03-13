from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual
from abctokz.config.schemas import TokenizerConfig
from abctokz.eval.metrics import fertility

PHRASE_1 = "आयो लाल, सभई चायो, झूलेलाल!"
PHRASE_2 = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"
PHRASES: list[tuple[str, str]] = [
    ("sindhi_phrase", PHRASE_1),
    ("marathi_phrase", PHRASE_2),
]

VOCAB_SIZES = [100, 400, 800]
MODEL_BUILDERS: dict[str, Callable[[int], TokenizerConfig]] = {
    "bpe": bpe_multilingual,
    "unigram": unigram_multilingual,
}

DEVANAGARI_RICH_CORPUS: list[str] = [
    PHRASE_1,
    PHRASE_2,
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "सारे जहाँ से अच्छा हिन्दोस्तां हमारा",
    "नमस्ते दुनिया",
    "भारत एक विशाल देश है",
    "हिन्दी भाषा में टोकनाइजेशन",
    "मराठी भाषेत टोकनायझेशन",
    "हे एक चाचणी वाक्य आहे",
    "मशीन लर्निंग के लिए टोकनाइज़ेशन उपयोगी है",
    "सिन्धी भाषा और देवनागरी लिपि",
    "उच्चारण और लेखन में अंतर हो सकता है",
    "देवनागरी में संयुक्ताक्षर महत्वपूर्ण होते हैं",
    "झूलेलाल सिंधी समाज में पूजनीय हैं",
    "गणपती बप्पा मोरया मंगलमूर्ति मोरया",
] * 30


def choose_runtime_tokenizer(trained: Tokenizer, loaded: Tokenizer, probe: str) -> tuple[Tokenizer, str]:
    a = trained.encode(probe)
    b = loaded.encode(probe)
    if a.tokens == b.tokens and a.ids == b.ids:
        return loaded, "loaded"
    return trained, "trained"


def train_tokenizer(model_name: str, vocab_size: int, corpus_path: Path, artifact_dir: Path) -> tuple[Tokenizer, str]:
    config = MODEL_BUILDERS[model_name](vocab_size=vocab_size)
    trained = Tokenizer.from_config(config)
    trained.train([str(corpus_path)], config)

    model_dir = artifact_dir / f"{model_name}_{vocab_size}"
    trained.save(str(model_dir))
    loaded = Tokenizer.load(str(model_dir))
    return choose_runtime_tokenizer(trained, loaded, probe="नमस्ते दुनिया")


def phrase_metrics(tokenizer: Tokenizer, text: str) -> dict[str, object]:
    enc = tokenizer.encode(text)
    words = len(text.split())
    f = fertility([enc], [words])
    return {
        "word_count": words,
        "token_count": len(enc.ids),
        "fertility": f,
        "tokens": enc.tokens,
        "ids": enc.ids,
    }


def harder_phrase(per_phrase: dict[str, dict[str, object]]) -> str:
    f1 = float(per_phrase["sindhi_phrase"]["fertility"])
    f2 = float(per_phrase["marathi_phrase"]["fertility"])
    if f1 > f2:
        return "sindhi_phrase"
    if f2 > f1:
        return "marathi_phrase"
    return "tie"


def main() -> None:
    artifact_root = Path("artifacts/task9")
    output_path = Path("outputs/task9/report.json")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_path = artifact_root / "devanagari_rich_corpus.txt"
    corpus_path.write_text("\n".join(DEVANAGARI_RICH_CORPUS), encoding="utf-8")

    results: dict[str, dict[str, object]] = {}

    for model_name in ["bpe", "unigram"]:
        model_result: dict[str, object] = {}
        for vs in VOCAB_SIZES:
            tok, source = train_tokenizer(model_name, vs, corpus_path, artifact_root)
            phrase_result: dict[str, dict[str, object]] = {}
            for label, text in PHRASES:
                phrase_result[label] = phrase_metrics(tok, text)
            model_result[str(vs)] = {
                "tokenizer_source": source,
                "phrases": phrase_result,
                "harder_phrase": harder_phrase(phrase_result),
            }
        results[model_name] = model_result

    report = {
        "phrases": {
            "sindhi_phrase": PHRASE_1,
            "marathi_phrase": PHRASE_2,
        },
        "vocab_sizes": VOCAB_SIZES,
        "models": results,
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 9: Measuring Phrase Difficulty")
    print(f"report: {output_path}")
    print()
    for model_name in ["bpe", "unigram"]:
        print(f"[{model_name}]")
        for vs in VOCAB_SIZES:
            entry = results[model_name][str(vs)]
            sp = entry["phrases"]["sindhi_phrase"]
            mp = entry["phrases"]["marathi_phrase"]
            print(
                f"  vocab={vs} | sindhi: tokens={sp['token_count']} fertility={sp['fertility']:.3f} "
                f"| marathi: tokens={mp['token_count']} fertility={mp['fertility']:.3f} "
                f"| harder={entry['harder_phrase']}"
            )
        print()


if __name__ == "__main__":
    main()
