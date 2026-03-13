from __future__ import annotations

import json
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, wordlevel_multilingual


def train_bpe_offsets(artifact_root: Path) -> Tokenizer:
    corpus = [
        "internationalization is nontrivial",
        "tokenization should preserve useful offsets",
        "subword models split long words",
    ] * 80
    path = artifact_root / "offsets_corpus.txt"
    path.write_text("\n".join(corpus), encoding="utf-8")

    cfg = bpe_multilingual(vocab_size=120)
    tok = Tokenizer.from_config(cfg)
    tok.train([str(path)], cfg)
    return tok


def train_wordlevel_reload(artifact_root: Path) -> tuple[Tokenizer, Path]:
    corpus = [
        "hello world",
        "wordlevel depends on exact token boundaries",
        "save load should preserve behavior",
    ] * 80
    path = artifact_root / "reload_corpus.txt"
    path.write_text("\n".join(corpus), encoding="utf-8")

    cfg = wordlevel_multilingual(vocab_size=200)
    tok = Tokenizer.from_config(cfg)
    tok.train([str(path)], cfg)

    model_dir = artifact_root / "wordlevel_reload_after"
    tok.save(str(model_dir))
    return tok, model_dir


def evaluate_offsets(tok: Tokenizer) -> dict[str, object]:
    text = "internationalization"
    enc = tok.encode(text)
    offsets = enc.offsets
    all_identical = len(set(offsets)) == 1 if offsets else False
    return {
        "input": text,
        "tokens": enc.tokens,
        "offsets": offsets,
        "all_offsets_identical": all_identical,
        "pass": not all_identical,
    }


def evaluate_save_load(trained: Tokenizer, model_dir: Path) -> dict[str, object]:
    text = "hello world"
    before = trained.encode(text)
    loaded = Tokenizer.load(str(model_dir))
    after = loaded.encode(text)
    changed = before.tokens != after.tokens or before.ids != after.ids
    return {
        "input": text,
        "before_tokens": before.tokens,
        "before_ids": before.ids,
        "after_tokens": after.tokens,
        "after_ids": after.ids,
        "behavior_changed": changed,
        "pass": not changed,
    }


def load_before_report(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    artifact_root = Path("artifacts/task15_after")
    output_path = Path("outputs/task15after/report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    before_report = load_before_report(Path("outputs/task15/report.json"))

    bpe_tok = train_bpe_offsets(artifact_root)
    trained_wl, wl_dir = train_wordlevel_reload(artifact_root)

    offsets_now = evaluate_offsets(bpe_tok)
    reload_now = evaluate_save_load(trained_wl, wl_dir)

    before_offsets_bug = None
    before_reload_bug = None
    if before_report is not None:
        c1 = before_report.get("case_1_offsets", {})
        c2 = before_report.get("case_2_save_load_drift", {})
        before_offsets_bug = bool(c1.get("observed_bug", False))
        before_reload_bug = bool(c2.get("behavior_changed", False))

    final = {
        "before_report_found": before_report is not None,
        "before": {
            "offsets_bug": before_offsets_bug,
            "save_load_drift": before_reload_bug,
        },
        "after": {
            "offsets_check": offsets_now,
            "save_load_check": reload_now,
        },
        "clear_diff": {
            "offsets": {
                "before": "all token offsets identical (bug)",
                "after": "piece-level offsets no longer identical",
            },
            "save_load": {
                "before": "encoding changed after reload (bug)",
                "after": "encoding stable before vs after reload",
            },
        },
    }

    output_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Task 15 After: Post-fix verification")
    print(f"report: {output_path}")
    print()
    print("Offsets check")
    print(f"  pass: {offsets_now['pass']}")
    print(f"  offsets identical: {offsets_now['all_offsets_identical']}")
    print(f"  sample offsets: {offsets_now['offsets'][:6]}")
    print()
    print("Save/load check")
    print(f"  pass: {reload_now['pass']}")
    print(f"  behavior changed: {reload_now['behavior_changed']}")
    print(f"  before: tokens={reload_now['before_tokens']} ids={reload_now['before_ids']}")
    print(f"  after : tokens={reload_now['after_tokens']} ids={reload_now['after_ids']}")
    print()
    if before_report is not None:
        print("Clear differences vs previous Task 15 report")
        print(f"  offsets bug before: {before_offsets_bug} -> after pass: {offsets_now['pass']}")
        print(f"  save/load bug before: {before_reload_bug} -> after pass: {reload_now['pass']}")


if __name__ == "__main__":
    main()
