"""
Task 7 — Does Encode → Decode Get You Back to Start?

Tests round-trip (encode then decode) across multilingual examples.
Demonstrates: one exact round-trip, one lossy round-trip (e.g. NFD vs NFC),
and what round_trip_success_rate actually measures.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from abctokz import Tokenizer
from abctokz.config.defaults import multilingual_shared_normalizer
from abctokz.config.schemas import (
    BPEConfig,
    BPETrainerConfig,
    DevanagariAwarePreTokenizerConfig,
    SequencePreTokenizerConfig,
    TokenizerConfig,
)
from abctokz.eval.metrics import round_trip_success_rate


# Small corpus to train a BPE tokenizer (mixed EN + Devanagari)
TRAIN_CORPUS: list[str] = [
    "hello world",
    "नमस्ते दुनिया",
    "encode decode round trip",
    "भारत एक विशाल देश है",
    "café and naïve",
    "देवनागरी लिपि",
] * 30


def build_config() -> TokenizerConfig:
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
        model=BPEConfig(unk_token="<unk>", vocab_size=1500),
        trainer=BPETrainerConfig(
            vocab_size=1500,
            min_frequency=2,
            special_tokens=["<unk>"],
        ),
    )


def main() -> None:
    out_dir = Path("outputs/task7")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("artifacts/task7")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = artifact_dir / "corpus.txt"
    corpus_path.write_text("\n".join(TRAIN_CORPUS), encoding="utf-8")

    config = build_config()
    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([str(corpus_path)], config)

    # -------------------------------------------------------------------------
    # Test cases: exact round-trip vs lossy (Unicode form change)
    # -------------------------------------------------------------------------

    # 1) Exact: ASCII / already NFC text — no normalizer change
    exact_sentences = [
        "hello world",
        "encode decode round trip",
        "नमस्ते दुनिया",  # already NFC in source
    ]

    # 2) Lossy (for string equality): NFD input → normalizer → NFC → encode → decode → NFC
    #    So decoded != original when original is NFD (they look the same on screen)
    nfd_cafe = "cafe\u0301"  # 'e' + combining acute (NFD)
    nfd_naive = unicodedata.normalize("NFD", "naïve")  # 'i' + combining diaeresis
    nfd_namaste = unicodedata.normalize("NFD", "नमस्ते")  # Devanagari in NFD
    lossy_sentences = [
        nfd_cafe + " world",
        "hello " + nfd_namaste,
        nfd_naive + " text",
    ]

    all_originals = exact_sentences + lossy_sentences
    encodings = tokenizer.encode_batch(all_originals)
    decoded = [tokenizer.decode(enc.ids) for enc in encodings]

    # Normalized versions (what the tokenizer actually sees after normalizer)
    normalized_originals = [unicodedata.normalize("NFC", s) for s in all_originals]

    # What round_trip_success_rate measures
    rate_vs_raw = round_trip_success_rate(all_originals, decoded)
    rate_vs_normalized = round_trip_success_rate(
        all_originals, decoded, normalized_originals=normalized_originals
    )

    # Build proof output
    proof = {
        "exact_round_trip_examples": [],
        "lossy_round_trip_examples": [],
        "nfd_vs_nfc": {},
        "round_trip_success_rate_vs_raw": rate_vs_raw,
        "round_trip_success_rate_vs_normalized": rate_vs_normalized,
        "metric_notes": (
            "round_trip_success_rate counts pairs where (target == decoded). "
            "With normalized_originals it compares decoded to NFC form; without, to raw input."
        ),
    }

    for i, (orig, dec) in enumerate(zip(exact_sentences, decoded[: len(exact_sentences)])):
        match = orig == dec
        proof["exact_round_trip_examples"].append({
            "original": orig,
            "decoded": dec,
            "round_trip_exact": match,
            "original_repr": repr(orig),
            "decoded_repr": repr(dec),
        })

    for i, (orig, dec, nfc) in enumerate(
        zip(lossy_sentences, decoded[len(exact_sentences) :], normalized_originals[len(exact_sentences) :])
    ):
        match_raw = orig == dec
        match_nfc = nfc == dec
        proof["lossy_round_trip_examples"].append({
            "original_repr": repr(orig),
            "decoded_repr": repr(dec),
            "round_trip_exact_vs_raw": match_raw,
            "round_trip_exact_vs_nfc": match_nfc,
            "explanation": "Input in NFD; normalizer converts to NFC before encode; decode outputs NFC.",
        })

    # Explicit NFD vs NFC: same logical text (use Latin to get distinct NFD form)
    nfc_text = "café"
    nfd_text = unicodedata.normalize("NFD", nfc_text)
    decoded_nfc = tokenizer.decode(tokenizer.encode(nfc_text).ids)
    decoded_nfd = tokenizer.decode(tokenizer.encode(nfd_text).ids)
    proof["nfd_vs_nfc"] = {
        "nfc_string": nfc_text,
        "nfd_string": nfd_text,
        "nfc_repr": repr(nfc_text),
        "nfd_repr": repr(nfd_text),
        "visually_identical": True,
        "codepoints_differ": nfc_text != nfd_text,
        "decoded_from_nfc_input": decoded_nfc,
        "decoded_from_nfd_input": decoded_nfd,
        "round_trip_nfc_exact": decoded_nfc == nfc_text,
        "round_trip_nfd_exact": decoded_nfd == nfd_text,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Console output for proofs
    print("Task 7 — Encode → Decode Round-Trip")
    print("=" * 60)
    print("\n1. Exact round-trip (original == decoded)")
    for ex in proof["exact_round_trip_examples"]:
        print(f"   original: {ex['original_repr']}")
        print(f"   decoded:  {ex['decoded_repr']}")
        print(f"   exact:    {ex['round_trip_exact']}")
    print("\n2. Lossy round-trip (NFD input → NFC decode)")
    for ex in proof["lossy_round_trip_examples"]:
        print(f"   original: {ex['original_repr']}")
        print(f"   decoded:  {ex['decoded_repr']}")
        print(f"   exact vs raw: {ex['round_trip_exact_vs_raw']} | vs NFC: {ex['round_trip_exact_vs_nfc']}")
        print(f"   → {ex['explanation']}")
    print("\n3. NFD vs NFC (same text, different Unicode form)")
    n = proof["nfd_vs_nfc"]
    print(f"   NFC repr: {n['nfc_repr']}")
    print(f"   NFD repr: {n['nfd_repr']}")
    print(f"   Visually identical: {n['visually_identical']} (display) | Codepoints differ: {n['codepoints_differ']}")
    print(f"   Decode(encode(NFC)) == NFC: {n['round_trip_nfc_exact']}")
    print(f"   Decode(encode(NFD)) == NFD: {n['round_trip_nfd_exact']}")
    print("\n4. round_trip_success_rate")
    print(f"   vs raw originals:       {rate_vs_raw:.4f}")
    print(f"   vs normalized (NFC):    {rate_vs_normalized:.4f}")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
