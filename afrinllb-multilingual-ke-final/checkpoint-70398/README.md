---
license: apache-2.0
tags:
- peft
- lora
- translation
- nllb
- nllb-200
languages:
- en
- sw
- ki
- kln
metrics:
- bleu
pipeline_tag: translation
---

# mcaaiNLLB

This is a fine-tuned LoRA (Low-Rank Adaptation) adapter for `facebook/nllb-200-distilled-600M` optimized for translating between English and Kenyan languages: Swahili, Kikuyu, and Kalenjin.

## Supported Languages & Tags

| Language Code | Language | NLLB Tag |
|---|---|---|
| `en` | English | `eng_Latn` |
| `sw` | Swahili | `swh_Latn` |
| `ki` | Kikuyu | `kik_Latn` |
| `kln` | Kalenjin | `kln_Latn` |

---

## How to Use (Google Colab / Python)

You can load and use this model for inference using `transformers` and `peft` with the following Python snippet.

### 1. Install Dependencies
```bash
pip install torch transformers peft accelerate
pip install --upgrade torchao
```

### 2. Run Inference
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# 1. Configuration
BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_REPO_ID = "MCAA1-MSU/mcaaiNLLB"

# Language tags map
NLLB_LANG_TAGS = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "ki": "kik_Latn",
    "kln": "kln_Latn"
}

# 2. Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading tokenizer and base model ({BASE_MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

print(f"Loading LoRA adapter ({ADAPTER_REPO_ID})...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_ID)
model.eval()
model.to(device)

# 3. Translation Helper function
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang not in NLLB_LANG_TAGS or tgt_lang not in NLLB_LANG_TAGS:
        raise ValueError(f"Unsupported language pair. Supported: {list(NLLB_LANG_TAGS.keys())}")
        
    tokenizer.src_lang = NLLB_LANG_TAGS[src_lang]
    tokenizer.tgt_lang = NLLB_LANG_TAGS[tgt_lang]
    target_lang_id = tokenizer.convert_tokens_to_ids(NLLB_LANG_TAGS[tgt_lang])
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 4. Example usage
if __name__ == "__main__":
    test_sentence = "Hello, how are you today?"
    
    print("\n--- Translation Examples ---")
    # English to Swahili
    sw_translation = translate(test_sentence, "en", "sw")
    print(f"EN: {test_sentence}")
    print(f"SW: {sw_translation}")
    
    # English to Kikuyu
    ki_translation = translate(test_sentence, "en", "ki")
    print(f"KI: {ki_translation}")
