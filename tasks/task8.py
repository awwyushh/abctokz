

from __future__ import annotations

from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.unicode_nfkc import NfkcNormalizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer

def main() -> None:
    print("=== Task 8: What Does the Normalizer Actually Do? ===")
    
    phrases = [
        ("Sindhi", "आयो लाल, सभई चायो, झूलेलाल!"),
        ("Marathi", "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!")
    ]
    
    dev_norm = DevanagariNormalizer()
    nfkc_norm = NfkcNormalizer()
    pre_tok = DevanagariAwarePreTokenizer(split_on_whitespace=True, split_on_script_boundary=True)
    
    for lang, text in phrases:
        print(f"\n--- {lang} Phrase ---")
        print(f"Raw Input:  {text!r}")
        
        normalized = dev_norm.normalize(text)
        print(f"Normalized: {normalized!r}")
        print(f"Identical?  {text == normalized}")
        
        pretokenized = pre_tok.pre_tokenize(normalized)
        print(f"Pre-tokenized: {pretokenized}")
        
    print("\n--- ZWJ / ZWNJ Preservation Test ---")
    conjunct_zwj = "क" + chr(0x094D) + chr(0x200D) + "ष" # क्‍ष
    print(f"Raw Input (with ZWJ): {conjunct_zwj!r} -> {[hex(ord(c)) for c in conjunct_zwj]}")
    print(f"Devanagari (NFC):   {dev_norm.normalize(conjunct_zwj)!r} -> {[hex(ord(c)) for c in dev_norm.normalize(conjunct_zwj)]}")
    print(f"NFKC (default):     {nfkc_norm.normalize(conjunct_zwj)!r} -> {[hex(ord(c)) for c in nfkc_norm.normalize(conjunct_zwj)]}")

if __name__ == "__main__":
    main()
