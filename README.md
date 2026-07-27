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
- **Premium Web UI:** Responsive, modern dashboard built with Gradio (hosted at the root path `/`) designed for desktop and mobile clients.
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

```bash
python app.py
```
On boot, the service:
1. Detects CUDA availability (and falls back to CPU if unavailable).
2. Downloads/loads `facebook/nllb-200-distilled-600M` and the LoRA adapter.
3. Quantizes and merges weights for optimal inference.
4. Mounts the Gradio user interface on `http://127.0.0.1:8000`.

---

## Docker Deployment

The application includes a `Dockerfile` and `docker-compose.yml` for containerized environments.

### Option A: Standard Build & Run (Docker CLI)

1. **Build the image:**
   ```bash
   docker build -t agri-translate-app:latest .
   ```

2. **Run the container (CPU Mode):**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate \
     agri-translate-app:latest
   ```

3. **Run the container (NVIDIA GPU Mode):**
   *(Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on host)*
   ```bash
   docker run -d \
     -p 8000:8000 \
     --gpus all \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate \
     agri-translate-app:latest
   ```

### Option B: Using Docker Compose (Recommended)

Docker Compose automatically configures container parameters and binds a persistent volume `hf_cache` on the host to avoid re-downloading model files (~5GB) when container resets.

1. **Uncomment GPU settings in `docker-compose.yml` (if using GPU):**
   Open `docker-compose.yml` and uncomment the `deploy` block under the services section:
   ```yaml
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
   ```

2. **Start the Stack:**
   ```bash
   docker-compose up -d --build
   ```

3. **Stop the Stack:**
   ```bash
   docker-compose down
   ```

---

## Production VPS / EC2 Deployment

Follow these steps to deploy the API to a remote Linux VPS (Ubuntu 20.04/22.04 LTS) e.g., on AWS EC2, DigitalOcean, or Linode.

### 1. Set Up the Application Service (systemd)

Creating a systemd service keeps the app running in the background and automatically restarts it if the server reboots or crashes.

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/agri-translate.service
   ```

2. Paste the following configuration, replacing `/path/to/NLLB_Model_API` with the absolute path of your workspace:
   ```ini
   [Unit]
   Description=AfriNLLB Translation & ASR Matrix API
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
   # Environment="HF_TOKEN=your_huggingface_token" # Uncomment if using private repos

   [Install]
   WantedBy=multi-user.target
   ```

3. Reload systemd, enable, and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable agri-translate.service
   sudo systemctl start agri-translate.service
   ```

4. Verify it is running properly:
   ```bash
   sudo systemctl status agri-translate.service
   journalctl -u agri-translate.service -n 50 -f
   ```

### 2. Configure Nginx Reverse Proxy with SSL

It is highly recommended to place the app behind a reverse proxy like Nginx to handle SSL certificates (HTTPS) and serve traffic securely on port 80/443.

1. Install Nginx:
   ```bash
   sudo apt update
   sudo apt install -y nginx
   ```

2. Create a new site configuration file:
   ```bash
   sudo nano /etc/nginx/sites-available/agri-translate
   ```

3. Paste the config block below (replace `api.yourdomain.com` with your real domain):
   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
           
           # Increase upload size limit for larger audio files
           client_max_body_size 50M;
           
           # Timeouts for heavy models
           proxy_read_timeout 300;
           proxy_connect_timeout 300;
           proxy_send_timeout 300;
       }
   }
   ```

4. Enable the configuration and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/agri-translate /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

5. Secure with SSL using Let's Encrypt (Certbot):
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d api.yourdomain.com
   ```
   *Follow the prompts to finalize the HTTPS configuration.*

---

## API Documentation

### 1. Interactive UI Client
- **URL:** `http://<your-server-ip>:8000/` (or `https://api.yourdomain.com/` in production)
- **Description:** A beautiful Gradio UI allowing clients to speak directly into their microphone or upload audio files, dynamically transcribing it and translating it.

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
    "translated_text": "Habari, umeamkaje leo?",
    "src_lang": "en",
    "tgt_lang": "sw"
  }
  ```
- **cURL Request Sample:**
  ```bash
  curl -X POST "http://localhost:8000/translate" \
       -H "Content-Type: application/json" \
       -d '{"text": "Hello, how are you today?", "src_lang": "en", "tgt_lang": "sw"}'
  ```

### 3. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response Sample:**
  ```json
  {
    "status": "healthy",
    "gpu_active": true
  }
  ```

### 4. Developer Interactive API Specs
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
