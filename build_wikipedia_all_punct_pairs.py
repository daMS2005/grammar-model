#!/usr/bin/env python3
import argparse, json, os, re, time, random
from typing import List, Dict
import requests
from tqdm import tqdm

# ---------------- Sentence splitting (NLTK if available, fallback otherwise) ----------------
_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+(?=[A-Z0-9(“"])')

def naive_split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def try_nltk_split(text: str) -> List[str]:
    try:
        import nltk
        try: nltk.data.find("tokenizers/punkt")
        except LookupError: nltk.download("punkt", quiet=True)
        try: nltk.data.find("tokenizers/punkt_tab")
        except Exception: pass
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return naive_split_sentences(text)

# ---------------- Punctuation handling (ALL punct, keep core cases) ----------------
WS = re.compile(r"\s+")
# ASCII + common Unicode punctuation
ALL_PUNCT_CHARS = r"""!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~“”„‟‘’‚‹›«»–—…"""
APOS_INWORD = re.compile(r"(?<=\w)'(?=\w)")   # keep it's, don't → do n't (we keep the original apostrophe)
DOT_INNUMBER = re.compile(r"(?<=\d)\.(?=\d)") # keep 3.14

def strip_all_punct_keep_core(s: str) -> str:
    s = APOS_INWORD.sub("<APOS>", s)
    s = DOT_INNUMBER.sub("<DOT>", s)
    s = re.sub(f"[{ALL_PUNCT_CHARS}]", " ", s)
    s = s.replace("<APOS>", "'").replace("<DOT>", ".")
    s = WS.sub(" ", s).strip()
    return s

def partial_strip_all_punct(s: str, p_drop: float = 0.6) -> str:
    """Randomly drop punctuation (except in-word apostrophes and decimal points)."""
    # Mask keepers
    s = APOS_INWORD.sub("<APOS>", s)
    s = DOT_INNUMBER.sub("<DOT>", s)
    out = []
    for ch in s:
        if re.match(f"[{ALL_PUNCT_CHARS}]", ch):
            # leave masked tokens as-is (they're not in ALL_PUNCT_CHARS)
            if random.random() < p_drop:
                out.append(" ")
            else:
                out.append(ch)
        else:
            out.append(ch)
    s2 = "".join(out)
    s2 = s2.replace("<APOS>", "'").replace("<DOT>", ".")
    return WS.sub(" ", s2).strip()

def has_any_punct(s: str) -> bool:
    return bool(re.search(f"[{ALL_PUNCT_CHARS}]", s))

def parens_balanced(s: str) -> bool:
    bal = 0
    for ch in s:
        if ch == "(": bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0: return False
    return bal == 0

def norm_words(x: str) -> str:
    # Compare words ignoring punctuation & case (preserve decimals and apostrophes first)
    x = APOS_INWORD.sub("<APOS>", x)
    x = DOT_INNUMBER.sub("<DOT>", x)
    x = re.sub(f"[{ALL_PUNCT_CHARS}]", " ", x)
    x = x.replace("<APOS>", "'").replace("<DOT>", ".")
    return WS.sub(" ", x).strip().lower()

# ---------------- Wikipedia fetch ----------------
def fetch_random_pages(lang: str, pages_needed: int, req_batch: int = 10, sleep_s: float = 0.15) -> List[Dict]:
    """
    Uses MediaWiki API to fetch random main-namespace pages with plaintext extracts.
    """
    req_batch = max(1, min(req_batch, 10))
    session = requests.Session()
    session.headers.update({"User-Agent": "PunctPairsBuilder/0.2 (contact: your-email@example.com)"})
    url = f"https://{lang}.wikipedia.org/w/api.php"

    pages = []
    with tqdm(total=pages_needed, desc="Fetching pages") as bar:
        while len(pages) < pages_needed:
            params = {
                "action": "query",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": min(req_batch, pages_needed - len(pages)),
                "prop": "extracts",
                "explaintext": 1,
                "format": "json"
            }
            try:
                r = session.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                if "query" in data and "pages" in data["query"]:
                    for pid, page in data["query"]["pages"].items():
                        title = page.get("title", "")
                        text = page.get("extract", "") or ""
                        if text.strip():
                            pages.append({"id": pid, "title": title, "text": text})
                            bar.update(1)
                time.sleep(sleep_s)
            except Exception:
                time.sleep(0.5)
                continue
    return pages

# ---------------- Pair building ----------------
def build_pairs_from_text(text: str, min_words: int, max_words: int,
                          mode: str, p_drop: float, mix: tuple) -> List[Dict]:
    """
    mode:
      - 'full'    : remove all punctuation
      - 'partial' : remove some punctuation
      - 'mixed'   : mix of full/partial/no-change using 'mix' ratios
      - 'noop'    : keep output as-is (teaches do-nothing)
    """
    out = []
    for s in try_nltk_split(text):
        if not s: continue
        wc = len(s.split())
        if not (min_words <= wc <= max_words): continue
        if not has_any_punct(s): continue               # ensure the target carries punctuation
        if not parens_balanced(s): continue             # keep sane targets

        if mode == "full":
            inp = strip_all_punct_keep_core(s)
        elif mode == "partial":
            inp = partial_strip_all_punct(s, p_drop=p_drop)
        elif mode == "noop":
            inp = s
        elif mode == "mixed":
            fs, ps, nc = mix
            r = random.random()
            if r < fs:
                inp = strip_all_punct_keep_core(s)
            elif r < fs + ps:
                inp = partial_strip_all_punct(s, p_drop=p_drop)
            else:
                inp = s
        else:
            raise ValueError("Unknown mode")

        # Must preserve words (ignoring punctuation)
        if norm_words(inp) != norm_words(s):
            continue
        # If this wasn't a 'noop' case, ensure some change happened
        if mode != "noop" and not (mode == "mixed" and r >= fs + ps) and inp == s:
            continue

        out.append({"input": inp, "output": s})
    return out

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Build (input, output) pairs from Wikipedia for ALL punctuation restoration.")
    ap.add_argument("--lang", default="en", help="Wikipedia language code")
    ap.add_argument("--pages", type=int, default=1200, help="How many random pages to fetch")
    ap.add_argument("--min-words", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--mode", choices=["full","partial","mixed","noop"], default="mixed",
                    help="full=remove all punct; partial=randomly drop some; mixed=blend; noop=teach do-nothing")
    ap.add_argument("--p-drop", type=float, default=0.6, help="Probability to drop a punctuation char in partial mode")
    ap.add_argument("--mix", type=str, default="0.45,0.45,0.10",
                    help="Ratios for mixed: full,partial,noop (sum≈1.0)")
    ap.add_argument("--max-pairs", type=int, default=5000, help="Stop after this many pairs")
    ap.add_argument("--eval-ratio", type=float, default=0.02, help="Fraction for eval split")
    ap.add_argument("--out-train", default="data/train.jsonl")
    ap.add_argument("--out-eval", default="data/eval.jsonl")
    ap.add_argument("--save-meta", action="store_true", help="Also save meta file with {title,url} per pair index")
    args = ap.parse_args()

    mix = tuple(float(x) for x in args.mix.split(","))
    assert len(mix) == 3 and 0.99 <= sum(mix) <= 1.01, "--mix must be three numbers summing to ~1.0"

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)

    pages = fetch_random_pages(args.lang, args.pages)
    pairs, meta = [], []
    for p in tqdm(pages, desc="Processing pages"):
        page_pairs = build_pairs_from_text(
            p["text"], args.min_words, args.max_words, args.mode, args.p_drop, mix
        )
        for ex in page_pairs:
            pairs.append(ex)
            meta.append({"title": p["title"], "url": f"https://{args.lang}.wikipedia.org/?curid={p['id']}"})
            if len(pairs) >= args.max_pairs:
                break
        if len(pairs) >= args.max_pairs:
            break

    # Deduplicate by normalized output text
    def norm_key(s: str) -> str:
        return WS.sub(" ", s.strip().lower())
    seen = set()
    dedup_pairs, dedup_meta = [], []
    for ex, m in zip(pairs, meta):
        key = norm_key(ex["output"])
        if key in seen: continue
        seen.add(key)
        dedup_pairs.append(ex); dedup_meta.append(m)

    random.shuffle(list(zip(dedup_pairs, dedup_meta)))
    n_eval = max(1, int(len(dedup_pairs) * args.eval_ratio)) if dedup_pairs else 0
    eval_set = dedup_pairs[:n_eval]
    train_set = dedup_pairs[n_eval:]
    eval_meta = dedup_meta[:n_eval]
    train_meta = dedup_meta[n_eval:]

    with open(args.out_train, "w", encoding="utf-8") as wt:
        for ex in train_set:
            wt.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_eval, "w", encoding="utf-8") as we:
        for ex in eval_set:
            we.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if args.save_meta:
        with open(args.out_train + ".meta.jsonl", "w", encoding="utf-8") as wm:
            for m in train_meta:
                wm.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(args.out_eval + ".meta.jsonl", "w", encoding="utf-8") as wm:
            for m in eval_meta:
                wm.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Pages fetched: {len(pages)}")
    print(f"Pairs built:   {len(pairs)}  → after dedupe: {len(dedup_pairs)}")
    print(f"Wrote {len(train_set)} train and {len(eval_set)} eval pairs to {args.out_train} / {args.out_eval}")
    print("License note: Wikipedia text is CC BY-SA 3.0; keep attribution if you redistribute (use --save-meta).")

if __name__ == "__main__":
    main()
