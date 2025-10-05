#!/usr/bin/env python3
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

def load_model(model_path: str):
    """Load the fine-tuned T5 model and tokenizer"""
    print(f"Loading model from {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def restore_punctuation(text: str, model, tokenizer, device, max_length: int = 512):
    """Restore punctuation for input text"""
    # Add task prefix
    input_text = f"restore punctuation: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="Test T5 punctuation restoration model")
    parser.add_argument("--model-path", default="./t5-punct-model", help="Path to fine-tuned model")
    parser.add_argument("--text", type=str, help="Text to restore punctuation for")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    if args.interactive:
        print("Interactive punctuation restoration mode. Type 'quit' to exit.")
        while True:
            text = input("\nEnter text without punctuation: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                result = restore_punctuation(text, model, tokenizer, device)
                print(f"Restored: {result}")
    
    elif args.text:
        result = restore_punctuation(args.text, model, tokenizer, device)
        print(f"Input:    {args.text}")
        print(f"Output:   {result}")
    
    else:
        # Test with some examples
        test_examples = [
            "hello world how are you today",
            "the quick brown fox jumps over the lazy dog",
            "i went to the store and bought some milk bread and eggs",
            "what time is it can you tell me please",
            "she said i dont know what youre talking about",
            "the meeting is scheduled for 3 30 pm on friday"
        ]
        
        print("Testing with example sentences:")
        for text in test_examples:
            result = restore_punctuation(text, model, tokenizer, device)
            print(f"Input:  {text}")
            print(f"Output: {result}")
            print()

if __name__ == "__main__":
    main() 