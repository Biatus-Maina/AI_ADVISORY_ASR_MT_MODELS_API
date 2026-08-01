# Swahili AgriTranslate API & Web Interface

A high-performance machine translation and Automatic Speech Recognition (ASR) service tailored for the agricultural domain. It hosts a fine-tuned LoRA adapter on `facebook/nllb-200-distilled-600M` optimized for translation between English and Kenyan languages (Swahili, Kikuyu, Kalenjin), alongside `microsoft/paza-whisper-large-v3-turbo` for transcribing spoken Swahili.

---

## Project Architecture

The application is structured as a **unified, memory-efficient service** that runs the FastAPI backend API and the Gradio user interface within a single Python process:

```
                  ┌───────────────────────────────┐
                  │          Public URL           │
                  │ (Ngrok/Cloudflare/Domain Host)│
                  └───────────────┬───────────────┘
                                  │ (port 8000)
                                  ▼
                  ┌───────────────────────────────┐
                  │        FastAPI App            │
                  ├───────────────────────────────┤
                  │  /docs   /health  /translate  │
                  └───────────────┬───────────────┘
                                  │ (mounted at /)
                                  ▼
                  ┌───────────────────────────────┐
                  │       Gradio Web UI           │
                  └───────────────┬───────────────┘
                                  │
      ┌───────────────────────────┴───────────────────────────┐
      ▼                                                       ▼
┌───────────────────────────┐                           ┌───────────────────────────┐
│     ASR Engine (GPU)      │                           │  Translation Engine (GPU) │
│ (paza-whisper-l3-turbo)   │                           │    (NLLB-2Dist + LoRA)    │
└───────────────────────────┘                           └───────────────────────────┘
```

### Key Technical Aspects:
- **Gradio Mounting:** The Gradio interface is mounted directly at the root (`/`) of the FastAPI application using `gr.mount_gradio_app`. This eliminates the need to run two separate Python instances, saving ~150MB of RAM and simplifying deployment.
- **GPU Acceleration & Automatic Fallback:** Heavy machine learning computations are executed on the GPU (`cuda`) using PyTorch. If CUDA is not available, the service automatically falls back to CPU execution with integer quantization to reduce footprint.
- **Persistent Model Caching:** Translation weights and Whisper models are cached locally to prevent redundant downloads on startup.
- **Decoupled API Logic:** The Gradio UI communicates internally with the API backend using local loopback calls (`http://127.0.0.1:8000/process-audio`).

---

## Interacting with the UI

The application serves a premium, responsive Web UI at the root path (`/`). It is designed to work on desktop and mobile browsers.

### UI Tabs & Features:

1. **Text Translation Tab:**
   - **How it works:** Input text, select your **Source Language** (e.g., Swahili, Kikuyu, Kalenjin, or English), select your **Target Language**, and click **Translate**.
   - **Use case:** Translating text-based agricultural advisories or chat messages.

2. **Audio Transcription & Translation Tab:**
   - **Microphone Input:** Press the **Record** button to record your voice directly from your device, and click **Stop** when finished.
   - **File Upload:** Alternatively, drag-and-drop or upload an existing audio file (`.wav`, `.mp3`, `.m4a`, etc.).
   - **Language Hints:** Specify the source language or select `Auto-detect` for Whisper to automatically identify the spoken language.
   - **Execution:** Click **Submit** to run the pipeline. The system will transcribe the spoken audio and translate it into your chosen target language.

---

## API Documentation

For integrations with external systems (such as automated SMS networks, agricultural apps, or advisory bots), developers can query the API programmatically.

### 1. Developer Interactive UI Specs
- **Swagger UI (Interactive Playground):** `https://your-domain.com/docs`
- **ReDoc (Static Documentation):** `https://your-domain.com/redoc`

### 2. Translation Endpoint
- **URL:** `/translate`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Parameters:**
  - `text` (string): The text to translate (minimum 1 character).
  - `src_lang` (string): Source language code (`en`, `sw`, `ki`, `kln`).
  - `tgt_lang` (string): Target language code (`en`, `sw`, `ki`, `kln`).
- **Request Sample:**
  ```json
  {
    "text": "Hello, how are you today?",
    "src_lang": "en",
    "tgt_lang": "sw"
  }
  ```
- **Response Sample:**
  ```json
  {
    "source_text": "Hello, how are you today?",
    "translated_text": "Halo, unaendeleaje leo?",
    "src_lang": "en",
    "tgt_lang": "sw"
  }
  ```

### 3. Speech Transcription & Translation Endpoint
- **URL:** `/process-audio`
- **Method:** `POST`
- **Headers:** `Content-Type: multipart/form-data`
- **Request Parameters:**
  - `file` (binary file): The audio file to process.
  - `src_lang` (string, Form field): The language spoken in the audio (`en`, `sw`, or `auto`).
  - `tgt_lang` (string, Form field): The language to translate the transcript to (`en`, `sw`, `ki`, `kln`).
- **Response Sample:**
  ```json
  {
    "transcript": "habari za asubuhi",
    "translation": "good morning"
  }
  ```

### 4. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response Sample:**
  ```json
  {
    "status": "healthy",
    "gpu_active": true
  }
  ```
