import torch
import static_ffmpeg
static_ffmpeg.add_paths()

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from contextlib import asynccontextmanager


# CONFIGURATION & STATE MANIFEST
BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"
FINAL_ADAPTER_PATH = "MCAA1-MSU/mcaaiNLLB"
ASR_MODEL_NAME = "microsoft/paza-whisper-large-v3-turbo"

NLLB_LANG_TAGS = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "ki": "kik_Latn",
    "kln": "kln_Latn"
}

# Persistent shared system RAM/VRAM containers
ml_models = {}

# LIFESPAN MANAGEMENT (Atomic Boot/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Guarantees the translation model, its adapters, and the ASR model are loaded
    into RAM/VRAM exactly once on boot, and cleanly flushed on API termination.
    """
    print("🚀 Initializing system infrastructure. Loading NLLB-LoRA weights into memory...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load NLLB Model & Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, FINAL_ADAPTER_PATH)
        model.eval()
        model.to(device)
        
        # Pin to application global scope state
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        ml_models["device"] = device
        print("✅ Core translation architectures successfully loaded.")
        
        # Load Whisper ASR model
        print("🎙️ Loading Whisper ASR model (microsoft/paza-whisper-large-v3-turbo)...")
        from transformers import pipeline
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            torch_dtype=torch_dtype,
            device=device_str
        )
        ml_models["asr_pipeline"] = asr_pipeline
        print("✅ Whisper ASR model successfully loaded.")
        
        print("✅ All model pipelines online.")
    except Exception as e:
        print(f"❌ Failed to initialize weights: {str(e)}")
        raise SystemExit(e)
        
    yield
    # Cleanup phase when server stops
    print("Shutting down... Purging runtime memory allocations.")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Helper function to translate text using the loaded NLLB-LoRA model.
    """
    if src_lang not in NLLB_LANG_TAGS or tgt_lang not in NLLB_LANG_TAGS:
        raise ValueError(f"Unsupported language pairing. Available codes: {list(NLLB_LANG_TAGS.keys())}")
        
    tokenizer = ml_models.get("tokenizer")
    model = ml_models.get("model")
    device = ml_models.get("device")
    
    if not tokenizer or not model:
        raise RuntimeError("ML Models are not loaded. Ensure lifespan startup completed successfully.")

    # Dynamically set runtime language target paths
    tokenizer.src_lang = NLLB_LANG_TAGS[src_lang]
    tokenizer.tgt_lang = NLLB_LANG_TAGS[tgt_lang]
    target_lang_id = tokenizer.convert_tokens_to_ids(NLLB_LANG_TAGS[tgt_lang])
    
    # Tokenize incoming request text payload
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Execute forward decoding pass
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        
    translated_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_output.strip()

# Initialize FastAPI App
app = FastAPI(
    title="AfriNLLB Multilingual Translation Matrix API",
    description="High-performance API engine serving fine-tuned LoRA translation matrices for Kenyan Languages.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend application web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, swap "*" for specific trusted web domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST DATA STRUCT VALIDATORS (Pydantic)
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Hello, how are you today?")
    src_lang: str = Field(..., example="en", description="Supported: en, sw, ki, kln")
    tgt_lang: str = Field(..., example="sw", description="Supported: en, sw, ki, kln")

class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    src_lang: str
    tgt_lang: str

# ENDPOINT CONTRACTS
@app.post("/translate", response_model=TranslationResponse)
async def perform_translation(payload: TranslationRequest):
    try:
        translated_output = translate_text(payload.text, payload.src_lang, payload.tgt_lang)
        return TranslationResponse(
            source_text=payload.text,
            translated_text=translated_output,
            src_lang=payload.src_lang,
            tgt_lang=payload.tgt_lang
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution engine failure: {str(e)}")

@app.get("/health")
async def health_check():
    """System heartbeat verification indicator."""
    return {"status": "healthy", "gpu_active": torch.cuda.is_available()}


# Gradio processing logic
# Gradio processing logic
def process_audio(audio_path, language_choice="Auto-detect"):
    if not audio_path:
        return "Tafadhali weka sauti.", ""
    
    try:
        asr_pipeline = ml_models.get("asr_pipeline")
        if not asr_pipeline:
            return "Kosa: Whisper model haijapakiwa.", ""
        
        print(f"🎙️ Transcribing: {audio_path} with language hint: {language_choice}")
        
        # Configure generation kwargs based on user choice or auto-detection
        generate_kwargs = {
            "task": "transcribe",
            "no_repeat_ngram_size": 4
        }
        if language_choice == "Swahili":
            generate_kwargs["language"] = "sw"
        elif language_choice == "English":
            generate_kwargs["language"] = "en"
        # If "Auto-detect", Whisper identifies the spoken language automatically

        result = asr_pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True
        )
        transcript_text = result["text"]
        
        # Strip any Whisper special tokens (e.g. <|transcribe|>, <|notimestamps|>, etc.)
        import re
        transcript_text = re.sub(r"<\|.*?\|>", "", transcript_text).strip()
        
        if not transcript_text or not transcript_text.strip():
            return "Haikupata maneno yoyote.", ""
            
        print(f"📝 Transcript: {transcript_text}")
        
        # Perform dynamic translation based on language detection
        english_words = {"the", "and", "is", "you", "to", "a", "i", "it", "in", "that", "was", "for", "on", "are", "as", "with", "his", "they", "at"}
        words = set(transcript_text.lower().split())
        is_english = len(words.intersection(english_words)) > 0 or language_choice == "English"
        
        if is_english:
            translated_text = translate_text(transcript_text, src_lang="en", tgt_lang="sw")
            print(f"🇸🇿 Swahili translation: {translated_text}")
        else:
            translated_text = translate_text(transcript_text, src_lang="sw", tgt_lang="en")
            print(f"🇺🇸 English translation: {translated_text}")
            
        return transcript_text, translated_text
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return f"Kosa wakati wa kuchakata: {str(e)}", ""


# Build Premium Gradio Interface
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

css_styles = """
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #10b981, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem !important;
    margin-bottom: 2px;
}
.sub-header {
    text-align: center;
    color: #64748b;
    font-size: 1.2rem;
    margin-bottom: 30px;
    font-weight: 500;
}
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    padding: 30px !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
}
"""

with gr.Blocks(title="Swahili AgriTranslate") as demo:
    gr.HTML("<h1 class='main-header'>Swahili AgriTranslate</h1>")
    gr.HTML("<p class='sub-header'>Swahili Speech Transcription & English Advisory Translation</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Sema au Weka Sauti (Record or Upload Audio)"
            )
            language_choice = gr.Dropdown(
                choices=["Auto-detect", "Swahili", "English"],
                value="Auto-detect",
                label="Target Language Hint"
            )
            with gr.Row():
                submit_btn = gr.Button("Tafsiri (Translate)", variant="primary")
                clear_btn = gr.Button("Futa (Clear)", variant="secondary")
            
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(
                label="Maandishi ya Sauti (Audio Transcript)",
                placeholder="Maandishi yataonekana hapa...",
                interactive=False,
                buttons=["copy"]
            )
            translation_output = gr.Textbox(
                label="Tafsiri (Translation)",
                placeholder="Tafsiri itaonekana hapa...",
                interactive=False,
                buttons=["copy"]
            )
            
    # Interactive wiring
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language_choice],
        outputs=[transcript_output, translation_output]
    )
    clear_btn.click(
        fn=lambda: (None, "Auto-detect", "", ""),
        inputs=None,
        outputs=[audio_input, language_choice, transcript_output, translation_output]
    )

# Mount Gradio sub-app at the root path of FastAPI
app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=theme,
    css=css_styles
)


if __name__ == "__main__":
    import uvicorn
    import os
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    # Disable reload by default to avoid reloading double models on CPU which takes too long
    uvicorn.run("app:app", host=host, port=port, reload=False)