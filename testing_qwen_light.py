#!/usr/bin/env python3
import argparse, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

def load_model(model_name, device, use_4bit, no_cache):
    kwargs = {}
    # Device & dtype
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
    elif device == "mps" and torch.backends.mps.is_available():
        dtype = torch.float16        # safer on MPS
        device_map = {"": "mps"}
    else:
        device = "cpu"
        dtype = torch.float32
        device_map = {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # Disabling cache reduces large temp allocations on MPS
    if no_cache:
        model.config.use_cache = False
    return tok, model, device

def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, messages, max_new_tokens, temperature, top_p):
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature is not None and temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    t = Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()
    out = ""
    for token in streamer:
        sys.stdout.write(token); sys.stdout.flush()
        out += token
    print()
    t.join()
    return out.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new", type=int, default=128)     # safer default
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--q4", action="store_true")
    ap.add_argument("--device", choices=["auto","cpu","mps","cuda"], default="auto")
    ap.add_argument("--no-cache", action="store_true", help="Disable KV cache (use on MPS if it crashes)")
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model: {args.model} on {dev} ...")
    tokenizer, model, dev = load_model(args.model, dev, args.q4, args.no_cache)
    print("Loaded. Type your message. Commands: /reset, /system <text>, /exit")

    messages = [{"role": "system", "content": args.system}]
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if not user: continue
        if user in {"/exit", ":q", ":quit"}: print("Goodbye!"); break
        if user.startswith("/system "):
            args.system = user[len("/system "):].strip()
            messages = [{"role": "system", "content": args.system}]
            print("(system prompt updated)"); continue
        if user == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("(history cleared)"); continue

        messages.append({"role": "user", "content": user})
        print("Assistant: ", end="", flush=True)
        reply = generate(model, tokenizer, messages, args.max_new, args.temp, args.top_p)
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
