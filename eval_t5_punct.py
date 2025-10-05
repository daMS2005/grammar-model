#!/usr/bin/env python3
import argparse, json, os, math, re, sys
from typing import List, Dict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---- helpers ---------------------------------------------------------------

WS = re.compile(r"\s+")
def norm_words(x: str) -> str:
    # remove commas/parentheses, collapse spaces, lowercase
    x = x.replace(",", " ").replace("(", " ").replace(")", " ")
    x = WS.sub(" ", x).strip().lower()
    return x

def count_punct(x: str) -> Dict[str,int]:
    return {
        ",": x.count(","),
        "(": x.count("("),
        ")": x.count(")"),
    }

def parens_balanced(x: str) -> bool:
    bal = 0
    for ch in x:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return False
    return bal == 0

def punct_prf(pred: str, ref: str):
    # position-agnostic counts (quick proxy)
    cp, cr = count_punct(pred), count_punct(ref)
    tp_commas = min(cp[","], cr[","])
    tp_op = min(cp["("], cr["("])
    tp_cl = min(cp[")"], cr[")"])
    pred_total = sum(cp.values())
    ref_total  = sum(cr.values())
    tp_total   = tp_commas + tp_op + tp_cl

    def prf(tp, p, r):
        prec = tp / p if p else 0.0
        rec  = tp / r if r else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return prec, rec, f1

    return {
        "all":  prf(tp_total, pred_total, ref_total),
        "comma":prf(tp_commas, cp[","], cr[","]),
        "open": prf(tp_op,     cp["("], cr["("]),
        "close":prf(tp_cl,     cp[")"], cr[")"]),
    }

# ---- main eval -------------------------------------------------------------

def load_model(model_name: str, device: str):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    if device == "mps":  # safer on Apple Metal
        model = model.to(dtype)
    return tok, model, device

def generate_batch(model, tok, inputs: List[str], device: str, max_new_tokens=128):
    enc = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic eval
        )
    outs = tok.batch_decode(out_ids, skip_special_tokens=True)
    return [o.strip() for o in outs]

def eval_dataset(preds: List[str], refs: List[str]) -> Dict[str, float]:
    assert len(preds) == len(refs)
    n = len(preds)
    exact = 0
    word_ident = 0
    paren_bal = 0

    # aggregate punct metrics
    agg = {"all_p":0.0,"all_r":0.0,"all_f":0.0,
           "comma_p":0.0,"comma_r":0.0,"comma_f":0.0,
           "open_p":0.0,"open_r":0.0,"open_f":0.0,
           "close_p":0.0,"close_r":0.0,"close_f":0.0}

    for y, ref in zip(preds, refs):
        if y == ref: exact += 1
        if norm_words(y) == norm_words(ref): word_ident += 1
        if parens_balanced(y): paren_bal += 1
        m = punct_prf(y, ref)
        agg["all_p"]   += m["all"][0];   agg["all_r"]   += m["all"][1];   agg["all_f"]   += m["all"][2]
        agg["comma_p"] += m["comma"][0]; agg["comma_r"] += m["comma"][1]; agg["comma_f"] += m["comma"][2]
        agg["open_p"]  += m["open"][0];  agg["open_r"]  += m["open"][1];  agg["open_f"]  += m["open"][2]
        agg["close_p"] += m["close"][0]; agg["close_r"] += m["close"][1]; agg["close_f"] += m["close"][2]

    def avg(x): return x/n if n else 0.0
    report = {
        "samples": n,
        "exact_match": exact/n if n else 0.0,
        "word_identity_ignoring_punct": word_ident/n if n else 0.0,
        "paren_balance_rate": paren_bal/n if n else 0.0,
        "punct_all_precision": avg(agg["all_p"]),
        "punct_all_recall":    avg(agg["all_r"]),
        "punct_all_f1":        avg(agg["all_f"]),
        "comma_precision":     avg(agg["comma_p"]),
        "comma_recall":        avg(agg["comma_r"]),
        "comma_f1":            avg(agg["comma_f"]),
        "open_paren_precision":avg(agg["open_p"]),
        "open_paren_recall":   avg(agg["open_r"]),
        "open_paren_f1":       avg(agg["open_f"]),
        "close_paren_precision":avg(agg["close_p"]),
        "close_paren_recall":   avg(agg["close_r"]),
        "close_paren_f1":       avg(agg["close_f"]),
    }
    return report

def main():
    ap = argparse.ArgumentParser(description="Evaluate a pretrained encoder–decoder on punctuation restoration (no finetune).")
    ap.add_argument("--model", default="google/byt5-small", help="HF model ID (e.g., google/byt5-small, google/flan-t5-base)")
    ap.add_argument("--data", default="", help="Path to JSONL with fields: input, output")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", choices=["auto","cpu","mps","cuda"], default="auto")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    tok, model, device = load_model(args.model, args.device)
    print(f"Loaded {args.model} on {device}")

    # Load data or fallback demo
    inputs, refs = [], []
    if args.data and os.path.exists(args.data):
        ds = load_dataset("json", data_files={"eval": args.data})["eval"]
        for ex in ds:
            if "input" in ex and "output" in ex and isinstance(ex["input"], str) and isinstance(ex["output"], str):
                inputs.append("Add only commas and parentheses. Do not change words:\n" + ex["input"])
                refs.append(ex["output"])
    else:
        print("No data provided; using tiny built-in demo set.")
        raw = [
            {
                "input":"i think however we should test this tomorrow if possible",
                "output":"I think, however, we should test this tomorrow, if possible."
            },
            {
                "input":"the result as shown in figure 2 is stable",
                "output":"The result (as shown in Figure 2) is stable."
            },
            {
                "input":"please send it to ana maria not juan",
                "output":"Please send it to Ana Maria, not Juan."
            },
        ]
        for ex in raw:
            inputs.append("Add only commas and parentheses. Do not change words:\n" + ex["input"])
            refs.append(ex["output"])

    preds = []
    for i in range(0, len(inputs), args.batch):
        batch_inp = inputs[i:i+args.batch]
        preds.extend(generate_batch(model, tok, batch_inp, device, max_new_tokens=args.max_new))
        print(f"Generated {len(preds)}/{len(inputs)}", end="\r")

    print("\nScoring…")
    report = eval_dataset(preds, refs)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Optional: dump predictions next to refs for spot-checking
    out_path = "predictions.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for inp, y, ref in zip(inputs, preds, refs):
            f.write(f"IN:\n{inp}\n\nPRED:\n{y}\n\nREF:\n{ref}\n{'-'*60}\n")
    print(f"Wrote detailed predictions to {out_path}")

if __name__ == "__main__":
    main()
