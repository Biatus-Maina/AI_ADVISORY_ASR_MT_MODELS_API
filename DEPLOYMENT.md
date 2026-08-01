# Swahili AgriTranslate - Installation & Deployment Guide

This document contains instructions for setting up, running, and deploying the Swahili AgriTranslate translation and ASR application.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Production VPS / EC2 Deployment (systemd & Ngrok)](#production-vps--ec2-deployment-systemd--ngrok)
4. [Docker Deployment](#docker-deployment)
5. [Troubleshooting & Optimizations](#troubleshooting--optimizations)

---

## System Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows WSL2.
- **RAM:** 
  - **CPU Only:** Minimum 8 GB RAM (12 GB+ recommended to run translation + speech models concurrently).
  - **GPU Execution:** Minimum 8 GB VRAM (e.g., NVIDIA RTX 3060, Tesla T4, or better).
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

### 2. Initialize Virtual Environment & Install Dependencies

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configurations

The application supports the following environment variables:

| Environment Variable | Description | Default Value |
|---|---|---|
| `HOST` | Bind address for the FastAPI server | `127.0.0.1` |
| `PORT` | Listening port for the FastAPI server | `8000` |
| `HF_HOME` | Directory where Hugging Face models are cached | `~/.cache/huggingface` |
| `HF_TOKEN` | Optional Hugging Face Access Token | *None* |
| `PYTHONUNBUFFERED` | Disable Python output buffering (recommended for systemd logs) | *None* |

### 4. Running Locally
Run the combined FastAPI application:
```bash
python app.py
```
On startup, the server loads both the translation and ASR pipelines into memory. The FastAPI API endpoints and the mounted Gradio UI will both be accessible at:
- **Web UI & API Root:** `http://127.0.0.1:8000/`
- **Interactive Documentation:** `http://127.0.0.1:8000/docs`

---

## Production VPS / EC2 Deployment (systemd & Ngrok)

For hosting in production persistently behind a firewall (like AWS Security Groups), we run the application as a user-level `systemd` service and tunnel the port using a persistent **Ngrok** service. This avoids needing `sudo` privileges or exposing open ports directly to the internet.

### 1. Configure Lingering
Enable lingering for your Linux user so that user-level systemd services start automatically on boot and remain running after you SSH log out:
```bash
loginctl enable-linger $USER
```

### 2. Configure Ngrok
1. Sign up for a free account at [ngrok.com](https://ngrok.com/).
2. Fetch your **Authtoken** and claim your **Free Static Domain** (e.g. `yourname.ngrok-free.dev`) from the dashboard.
3. Install Ngrok to your local binary path:
   ```bash
   mkdir -p ~/.local/bin
   wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   tar -xzf ngrok-v3-stable-linux-amd64.tgz -C ~/.local/bin/
   rm ngrok-v3-stable-linux-amd64.tgz
   ```
4. Authenticate the agent:
   ```bash
   ~/.local/bin/ngrok config add-authtoken <YOUR_AUTHTOKEN>
   ```

### 3. Create systemd Service Configurations

Create the user systemd service directory:
```bash
mkdir -p ~/.config/systemd/user/
```

#### A. Backend API Service
Create `~/.config/systemd/user/agri-translate-api.service`:
```ini
[Unit]
Description=AfriNLLB Translation & ASR Backend API
After=network.target

[Service]
WorkingDirectory=/home/biatus/AI_ADVISORY_ASR_MT_MODELS_API
ExecStart=/home/biatus/AI_ADVISORY_ASR_MT_MODELS_API/.venv/bin/python app.py
Restart=always
RestartSec=5
Environment="PORT=8000"
Environment="HOST=0.0.0.0"

[Install]
WantedBy=default.target
```

#### B. Ngrok Tunnel Service
Create `~/.config/systemd/user/agri-translate-ngrok.service`:
```ini
[Unit]
Description=Ngrok Tunnel for AgriTranslate Web UI & API
After=network.target agri-translate-api.service

[Service]
ExecStart=/home/biatus/.local/bin/ngrok http --url=<YOUR_STATIC_DOMAIN> 8000
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

### 4. Enable and Start Services
Reload the systemd user daemon, enable auto-start, and boot the services:
```bash
systemctl --user daemon-reload
systemctl --user enable agri-translate-api.service agri-translate-ngrok.service
systemctl --user start agri-translate-api.service agri-translate-ngrok.service
```

---

## Docker Deployment

The application includes `Dockerfile.api` and a `docker-compose.yml` to facilitate containerization.

### Option A: Standard Build & Run (Docker CLI)

1. **Build the API Image:**
   ```bash
   docker build -t agri-translate-api:latest -f Dockerfile.api .
   ```

2. **Run (CPU Mode):**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate-api \
     agri-translate-api:latest
   ```

3. **Run (NVIDIA GPU Mode):**
   *(Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host)*
   ```bash
   docker run -d \
     -p 8000:8000 \
     --gpus all \
     -v $(pwd)/hf_cache:/cache \
     -e HF_HOME=/cache \
     --name agri-translate-api \
     agri-translate-api:latest
   ```

### Option B: Docker Compose (Recommended)
Docker Compose handles persistent model volume mapping and port configurations automatically.

1. **Start Stack:**
   ```bash
   docker-compose up -d --build
   ```
2. **Stop Stack:**
   ```bash
   docker-compose down
   ```

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
