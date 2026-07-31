# Swahili AgriTranslate API & Web Interface

A high-performance machine translation and Automatic Speech Recognition (ASR) service tailored for the agricultural domain. It hosts a fine-tuned LoRA adapter on `facebook/nllb-200-distilled-600M` optimized for translation between English and Kenyan languages (Swahili, Kikuyu, Kalenjin), alongside `microsoft/paza-whisper-large-v3-turbo` for transcribing spoken Swahili.

The application serves a premium, responsive Gradio user interface at the root URL (`/`) and developer-friendly FastAPI endpoints (`/translate`, `/health`, `/docs`) for integration with external agricultural advisory systems.

---

## Table of Contents
1. [Features](#features)
2. [Supported Languages](#supported-languages)
3. [System Requirements](#system-requirements)
4. [Local Development Setup](#local-development-setup)
5. [Docker Deployment](#docker-deployment)
6. [Production VPS / EC2 Deployment](#production-vps--ec2-deployment)
7. [API Documentation](#api-documentation)
8. [Troubleshooting & Optimizations](#troubleshooting--optimizations)

---

## Features
- **Integrated Speech-to-Text & Translation:** End-to-end processing of audio inputs (microphone or files) to transcribe Swahili or English and translate them dynamically.
- **Web UI:** Responsive, modern dashboard built with Gradio (hosted at the root path `/`) designed for desktop and mobile clients.
- **FastAPI Endpoints:** Fast, asynchronous, type-safe API endpoints for backend integration.
- **Resource Management:** Memory-optimized loading of weights, pinning automatically between GPU (CUDA) and CPU, with CPU quantization to reduce footprint.
- **Persistent Cache Mapping:** Keeps Hugging Face models cached locally or inside container volumes to avoid redundant downloads.

---

## Supported Languages

The NLLB model translator supports the following mappings:

| Language | Code | NLLB Target Tag |
|---|---|---|
| English | `en` | `eng_Latn` |
| Swahili | `sw` | `swh_Latn` |
| Kikuyu | `ki` | `kik_Latn` |
| Kalenjin | `kln` | `kln_Latn` |

---

## System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows WSL2.
- **RAM:** 
  - **CPU Only:** Minimum 8 GB RAM (12 GB+ recommended to run translation + speech models concurrently).
  - **GPU Execution:** Minimum 8 GB VRAM (e.g., NVIDIA RTX 3060, T4, or better).
- **System Packages:** `ffmpeg` and `libsndfile1` are required on the host system to process audio.

---

## Local Development Setup

### 1. Install System Dependencies

Before running the application, make sure `ffmpeg` and `libsndfile1` are installed.

* **Ubuntu / Debian:**
  ```bash
  sudo apt update
  sudo apt install -y ffmpeg libsndfile1
  ```
* **macOS:**
  ```bash
  brew install ffmpeg libsndfile
  ```
* **Windows (PowerShell with Chocolatey):**
  ```powershell
  choco install ffmpeg
  ```

### 2. Clone and Initialize Virtual Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configurations (Optional)

The application recognizes the following environment variables:

| Environment Variable | Description | Default Value |
|---|---|---|
| `HOST` | Bind address for the server | `127.0.0.1` |
| `PORT` | Listening port for the application | `8000` |
| `HF_HOME` | Directory where models are cached | `~/.cache/huggingface` |
| `HF_TOKEN` | Hugging Face Access Token (improves API limits) | *None* |

### 4. Run the Application

Since the API backend and Gradio UI frontend are separated, you must start both services:

- **Start the FastAPI Backend Service:**
  ```bash
  python app.py
  ```
  On boot, the service loads the ML models (NLLB + Whisper ASR) into memory and runs on `http://127.0.0.1:8000`.

- **Start the Gradio UI Client:**
  ```bash
  python ui.py
  ```
  The Gradio frontend runs on `http://127.0.0.1:8080`. By default, it communicates with the local API at `http://127.0.0.1:8000`. You can configure the API endpoint using the `API_URL` environment variable:
  ```bash
  API_URL=http://your-remote-api-ip:8000 python ui.py
  ```

---

## Docker Deployment

The application uses two separate Dockerfiles to enable optimized containerized setups (e.g. keeping the UI container extremely small and resource-efficient).

### Option A: Standard Build & Run (Docker CLI)

1. **Build the images:**
   ```bash
   # Build the API Backend
   docker build -t agri-translate-api:latest -f Dockerfile.api .

   # Build the Gradio UI Client
   docker build -t agri-translate-ui:latest -f Dockerfile.ui .
   ```

2. **Run the API Backend (CPU Mode):**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate-api \
     agri-translate-api:latest
   ```

3. **Run the API Backend (NVIDIA GPU Mode):**
   *(Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on host)*
   ```bash
   docker run -d \
     -p 8000:8000 \
     --gpus all \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate-api \
     agri-translate-api:latest
   ```

4. **Run the Gradio UI Frontend:**
   ```bash
   docker run -d \
     -p 8080:8080 \
     -e API_URL=http://<host-ip-address>:8000 \
     --name agri-translate-ui \
     agri-translate-ui:latest
   ```

### Option B: Using Docker Compose (Recommended)

Docker Compose starts both containers automatically and maps their internal networks so they can communicate seamlessly. It also configures a persistent volume `hf_cache` on the host to avoid re-downloading model files (~5GB).

1. **Uncomment GPU settings in `docker-compose.yml` (if using GPU):**
   Open `docker-compose.yml` and uncomment the `deploy` block under the `agri-translate-api` service.

2. **Start the Stack:**
   ```bash
   docker-compose up -d --build
   ```
   The UI will be accessible on port `8080` (`http://localhost:8080`) and the API will be accessible on port `8000` (`http://localhost:8000`).

3. **Stop the Stack:**
   ```bash
   docker-compose down
   ```

---

## Production VPS / EC2 Deployment

Because the services are decoupled, they can be deployed on different machines. For example, you can host the heavy machine learning models on a GPU-enabled instance, and host the Gradio UI on a small, cheap CPU instance.

### 1. Set Up the Application Service (systemd)

Creating a systemd service keeps the backend running automatically.

1. Create a service file for the API:
   ```bash
   sudo nano /etc/systemd/system/agri-translate-api.service
   ```

2. Paste the configuration below (replace `/home/ubuntu/NLLB_Model_API` with your workspace directory path):
   ```ini
   [Unit]
   Description=AfriNLLB Translation & ASR Backend API
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/NLLB_Model_API
   ExecStart=/home/ubuntu/NLLB_Model_API/.venv/bin/python app.py
   Restart=always
   RestartSec=5
   Environment="PORT=8000"
   Environment="HOST=127.0.0.1"
   Environment="HF_HOME=/home/ubuntu/.cache/huggingface"

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable agri-translate-api.service
   sudo systemctl start agri-translate-api.service
   ```

4. Repeat steps to run the UI service (`agri-translate-ui.service`) using `/home/ubuntu/NLLB_Model_API/.venv/bin/python ui.py` with `PORT=8080` and `API_URL=http://127.0.0.1:8000`.

---

## API Documentation

### 1. Interactive UI Client
- **URL:** `http://<your-server-ip>:8080/` (or `https://yourdomain.com/` in production)
- **Description:** A beautiful Gradio UI allowing clients to record audio via their microphone or upload audio files, dynamically transcribing it and translating it.

### 2. Translation Endpoint
- **URL:** `/translate`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Parameters:**
  - `text` (string): The text to translate. Must be at least 1 character.
  - `src_lang` (string): The source language code (choices: `en`, `sw`, `ki`, `kln`).
  - `tgt_lang` (string): The target language code (choices: `en`, `sw`, `ki`, `kln`).
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

### 3. Speech Transcription & Translation Endpoint (Gradio Backend)
- **URL:** `/process-audio`
- **Method:** `POST`
- **Headers:** `Content-Type: multipart/form-data`
- **Request Parameters:**
  - `file` (binary): The audio file to process.
  - `language_choice` (string, Form parameter): Optional language choice/hint. Defaults to `"Auto-detect"` (options: `"Auto-detect"`, `"Swahili"`, `"English"`).
- **Response Sample:**
  ```json
  {
    "transcript": "Habari za asubuhi",
    "translation": "Good morning"
  }
  ```

### 4. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response Sample:**
  ```json
  {
    "status": "healthy",
    "gpu_active": false
  }
  ```

### 5. Developer Interactive API Specs
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Troubleshooting & Optimizations

### Local vs Hugging Face Hub Adapter
By default, the application pulls the adapter matrix from Hugging Face Hub: `MCAA1-MSU/mcaaiNLLB`. 
If you want to use the local checkpoint stored inside the repository:
1. Open `app.py`
2. Modify `FINAL_ADAPTER_PATH`:
   ```python
   # From:
   FINAL_ADAPTER_PATH = "MCAA1-MSU/mcaaiNLLB"
   # To:
   FINAL_ADAPTER_PATH = "./afrinllb-multilingual-ke-final/checkpoint-70398"
   ```

### CPU Memory Optimization
Loading Whisper and NLLB simultaneously can lead to out-of-memory errors on smaller systems. 
- Ensure you have at least 8 GB of swap space configured if running on a 4 GB RAM/VRAM VPS.
- If you notice heavy lag during model downloads, verify that Nginx timeouts are configured to `300s` or higher as models download sequentially on the first boot.

