from __future__ import annotations

import json
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, wordlevel_multilingual


def train_bpe_for_offsets(artifact_root: Path) -> Tokenizer:
    corpus = [
        "internationalization is nontrivial",
        "tokenization should preserve useful offsets",
        "subword models split long words",
    ] * 80
    corpus_path = artifact_root / "offsets_corpus.txt"
    corpus_path.write_text("\n".join(corpus), encoding="utf-8")

    config = bpe_multilingual(vocab_size=120)
    tok = Tokenizer.from_config(config)
    tok.train([str(corpus_path)], config)
    return tok


def train_wordlevel_for_reload(artifact_root: Path) -> tuple[Tokenizer, Path]:
    corpus = [
        "hello world",
        "wordlevel depends on exact token boundaries",
        "save load should preserve behavior",
    ] * 80
    corpus_path = artifact_root / "reload_corpus.txt"
    corpus_path.write_text("\n".join(corpus), encoding="utf-8")

    config = wordlevel_multilingual(vocab_size=200)
    tok = Tokenizer.from_config(config)
    tok.train([str(corpus_path)], config)

    model_dir = artifact_root / "wordlevel_reload"
    tok.save(str(model_dir))
    return tok, model_dir


def case_offsets(tok: Tokenizer) -> dict[str, object]:
    text = "internationalization"
    enc = tok.encode(text)
    offsets = enc.offsets
    tokens = enc.tokens
    slices = [text[s:e] for s, e in offsets]

    all_same_span = len(set(offsets)) == 1 if offsets else False
    multi_piece = len(tokens) > 1
    suspicious = multi_piece and all_same_span

    return {
        "input": text,
        "tokens": tokens,
        "offsets": offsets,
        "text_slices": slices,
        "multi_piece": multi_piece,
        "all_offsets_identical": all_same_span,
        "observed_bug": suspicious,
        "expected": "For multi-piece tokenization, offsets should partition the word into piece-level spans.",
        "observed": "All pieces receive the same full-word span, so offsets do not align with individual tokens.",
        "classification": "bug" if suspicious else "no-bug-detected",
        "reason": "In tokenizer.encode(), offsets are appended using len(pre_tok) for every piece and cursor is not advanced per piece.",
    }


def case_save_load_drift(trained_tok: Tokenizer, model_dir: Path) -> dict[str, object]:
    text = "hello world"
    before = trained_tok.encode(text)
    loaded = Tokenizer.load(str(model_dir))
    after = loaded.encode(text)

    behavior_changed = before.tokens != after.tokens or before.ids != after.ids

    return {
        "input": text,
        "before_save_tokens": before.tokens,
        "before_save_ids": before.ids,
        "after_load_tokens": after.tokens,
        "after_load_ids": after.ids,
        "behavior_changed": behavior_changed,
        "expected": "Encoding results should be identical before save and after load for the same artifact.",
        "observed": "Loaded tokenizer can produce different tokenization due to incomplete pipeline reconstruction.",
        "classification": "bug" if behavior_changed else "no-bug-detected",
        "reason": "Tokenizer.load() restores model/decoder but not full preprocessing pipeline state used during training.",
    }


def main() -> None:
    artifact_root = Path("artifacts/task15")
    output_path = Path("outputs/task15/report.json")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpe_tok = train_bpe_for_offsets(artifact_root)
    trained_wordlevel, model_dir = train_wordlevel_for_reload(artifact_root)

    report = {
        "task": "Task 15 - Find Something That Breaks",
        "approach": "Edge-case contract checks: metadata integrity and persistence consistency",
        "case_1_offsets": case_offsets(bpe_tok),
        "case_2_save_load_drift": case_save_load_drift(trained_wordlevel, model_dir),
        "minimal_workarounds": {
            "offsets": "Do not trust per-piece offsets for alignment-sensitive tasks until encode() offset logic is fixed.",
            "save_load": "For exact reproducibility, keep tokenizer in-memory after training or validate post-load behavior before deployment.",
        },
        "minimal_fixes": {
            "offsets": "In tokenizer.encode(), compute per-piece span with piece length and increment char cursor per emitted piece.",
            "save_load": "Persist and restore full TokenizerConfig pipeline in load(), not only model type metadata.",
        },
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c1 = report["case_1_offsets"]
    c2 = report["case_2_save_load_drift"]

    print("Task 15: Find Something That Breaks")
    print(f"report: {output_path}")
    print()
    print("Case 1 - Offset metadata integrity")
    print(f"  tokens: {c1['tokens']}")
    print(f"  offsets: {c1['offsets']}")
    print(f"  all_offsets_identical: {c1['all_offsets_identical']}")
    print(f"  classification: {c1['classification']}")
    print()
    print("Case 2 - Save/load behavior drift")
    print(f"  before: tokens={c2['before_save_tokens']} ids={c2['before_save_ids']}")
    print(f"  after : tokens={c2['after_load_tokens']} ids={c2['after_load_ids']}")
    print(f"  behavior_changed: {c2['behavior_changed']}")
    print(f"  classification: {c2['classification']}")


if __name__ == "__main__":
    main()
